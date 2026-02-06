/**
 * Surface Loader for 3D mesh rendering.
 *
 * Fetches mesh data from backend and renders as VTK actors in the volume3d viewport.
 * Bypasses OHIF segmentation API entirely — OHIF's addSegmentations() assumes
 * DICOM volume-based segmentations (imageIds required), which standalone meshes lack.
 */

import { reapplyLabelmapToViewports, getSegmentStates } from './onDemandSegmentationManager';

export const VOLUME3D_VIEWPORT_ID = 'volume3d';

interface MeshData {
  vertices: number[][];
  faces: number[][];
  color: number[];
  segmentIndex: number;
  vertexCount: number;
  faceCount: number;
}

interface SurfaceMeshResponse {
  studyUID: string;
  structureCount: number;
  structures: Record<string, MeshData>;
}

const addedActorUIDs = new Map<string, string[]>();
const meshDataCache = new Map<string, SurfaceMeshResponse>();

let layoutChangeUnsubscribe: (() => void) | null = null;
let currentServicesManager: any = null;
let currentStudyUID: string | null = null;

/**
 * Apply CT-Bone style rendering directly to a vtkVolume actor.
 * Bypasses viewport.setProperties({ preset }) which leaks to 2D viewports via shared volume state.
 */
function applyBonePresetToActor(volumeActor: any): void {
  const prop = volumeActor.getProperty();

  const cfun = prop.getRGBTransferFunction(0);
  cfun.removeAllPoints();
  cfun.addRGBPoint(-3024, 0, 0, 0);
  cfun.addRGBPoint(-16, 0.73, 0.25, 0.30);
  cfun.addRGBPoint(641, 0.91, 0.82, 0.56);
  cfun.addRGBPoint(3071, 1, 1, 1);

  prop.setShade(true);
  prop.setAmbient(0.1);
  prop.setDiffuse(0.9);
  prop.setSpecular(0.2);
  prop.setSpecularPower(10);
  prop.setUseGradientOpacity(0, false);
}

export function setVolume3DPreset(
  servicesManager: any,
  _presetName: string = 'CT-Bone'
): boolean {
  try {
    const viewportId = findVolume3DViewportId(servicesManager);
    if (!viewportId) return false;

    const { cornerstoneViewportService } = servicesManager.services;
    const viewport = cornerstoneViewportService?.getCornerstoneViewport?.(viewportId);
    if (!viewport) return false;

    const actors = viewport.getActors?.() || [];
    for (const actorEntry of actors) {
      const actor = actorEntry.actor || actorEntry;
      if (actor?.getClassName?.() === 'vtkVolume') {
        applyBonePresetToActor(actor);
        viewport.render();
        console.log('[SurfaceLoader] Volume3D bone preset applied directly to actor');
        return true;
      }
    }

    console.warn('[SurfaceLoader] No vtkVolume actor found in volume3d viewport');
    return false;
  } catch (e: any) {
    console.warn('[SurfaceLoader] Could not set volume preset:', e?.message || e);
    return false;
  }
}

export async function fetchSurfaceMeshes(studyUID: string): Promise<SurfaceMeshResponse | null> {
  const url = `/api/surface-mesh/${studyUID}?decimate=1.0`;
  console.log(`[SurfaceLoader] Fetching meshes from: ${url}`);

  try {
    const response = await fetch(url);

    if (!response.ok) {
      if (response.status === 404) {
        console.log('[SurfaceLoader] Mesh not available (run nerve analysis first)');
        return null;
      }
      throw new Error(`HTTP ${response.status}: ${response.statusText}`);
    }

    const data: SurfaceMeshResponse = await response.json();
    console.log(`[SurfaceLoader] Received ${data.structureCount} structures`);
    return data;
  } catch (error) {
    console.error('[SurfaceLoader] Failed to fetch meshes:', error);
    return null;
  }
}

function getCornerstoneLibs(): { cornerstone: any; cornerstoneTools: any } | null {
  const cornerstone = (window as any).cornerstone || (window as any).cornerstoneCore;
  const cornerstoneTools = (window as any).cornerstoneTools;

  if (!cornerstone || !cornerstoneTools) {
    console.error('[SurfaceLoader] Cornerstone libraries not found on window');
    return null;
  }

  return { cornerstone, cornerstoneTools };
}

function findVolume3DViewportId(servicesManager: any): string | null {
  try {
    const { cornerstoneViewportService } = servicesManager.services;
    const viewportIds = cornerstoneViewportService.getViewportIds?.() || [];

    for (const vpId of viewportIds) {
      const vpInfo = cornerstoneViewportService.getViewportInfo?.(vpId);
      if (vpInfo?.viewportOptions?.viewportType === 'volume3d') {
        return vpId;
      }
    }

    if (viewportIds.includes(VOLUME3D_VIEWPORT_ID)) {
      return VOLUME3D_VIEWPORT_ID;
    }
  } catch (e) {
    console.warn('[SurfaceLoader] Error finding volume3d viewport:', e);
  }

  return null;
}

export async function loadSurfaceMeshesToViewport(
  studyUID: string,
  servicesManager: any
): Promise<boolean> {
  console.log(`[SurfaceLoader] Loading surfaces for study: ${studyUID}`);

  const libs = getCornerstoneLibs();
  if (!libs) return false;

  const { cornerstone } = libs;

  const viewportId = findVolume3DViewportId(servicesManager);
  if (!viewportId) {
    console.error('[SurfaceLoader] No volume3d viewport found');
    return false;
  }

  const { cornerstoneViewportService } = servicesManager.services;
  const viewport = cornerstoneViewportService.getCornerstoneViewport?.(viewportId);
  if (!viewport) {
    console.error('[SurfaceLoader] Could not get volume3d viewport object');
    return false;
  }

  let meshData = meshDataCache.get(studyUID);
  if (!meshData) {
    meshData = await fetchSurfaceMeshes(studyUID);
    if (meshData) {
      meshDataCache.set(studyUID, meshData);
    }
  }

  if (!meshData || meshData.structureCount === 0) {
    console.log('[SurfaceLoader] No mesh data available');
    return false;
  }

  const { geometryLoader, cache, Enums: coreEnums } = cornerstone;

  if (!geometryLoader?.createAndCacheGeometry) {
    console.error('[SurfaceLoader] geometryLoader.createAndCacheGeometry not available');
    return false;
  }

  // frameOfReferenceUID required by cornerstone3D SurfaceData
  let frameOfReferenceUID = '';
  const vpIds = cornerstoneViewportService.getViewportIds?.() || [];
  for (const vpId of vpIds) {
    try {
      const vp = cornerstoneViewportService.getCornerstoneViewport?.(vpId);
      if (!vp) continue;
      const forUID = vp.getFrameOfReferenceUID?.();
      if (forUID) {
        frameOfReferenceUID = forUID;
        break;
      }
    } catch (_) {}
  }

  if (!frameOfReferenceUID) {
    console.error('[SurfaceLoader] Could not get frameOfReferenceUID');
    return false;
  }

  let vtkPolyData: any, vtkMapper: any, vtkActor: any;
  try {
    // @ts-ignore
    const polyDataMod = await import('@kitware/vtk.js/Common/DataModel/PolyData');
    // @ts-ignore
    const mapperMod = await import('@kitware/vtk.js/Rendering/Core/Mapper');
    // @ts-ignore
    const actorMod = await import('@kitware/vtk.js/Rendering/Core/Actor');
    vtkPolyData = polyDataMod.default || polyDataMod;
    vtkMapper = mapperMod.default || mapperMod;
    vtkActor = actorMod.default || actorMod;
  } catch (e) {
    console.error('[SurfaceLoader] Failed to import vtk.js:', e);
    return false;
  }

  if (!vtkPolyData?.newInstance || !vtkMapper?.newInstance || !vtkActor?.newInstance) {
    console.error('[SurfaceLoader] vtk.js not properly imported');
    return false;
  }

  const actorEntries: Array<{ uid: string; actor: any }> = [];
  const actorUIDs: string[] = [];

  try {
    let segmentIndex = 1;
    for (const [name, mesh] of Object.entries(meshData.structures)) {
      const geometryId = `surface-${studyUID}-${name}`;

      const points = new Float32Array(mesh.vertices.flat());

      // VTK connectivity format: [nPts, p0, p1, p2, ...]
      const flatFaces = mesh.faces.flat();
      const numTriangles = mesh.faceCount;
      const polysArray = new Uint32Array(numTriangles * 4);
      for (let i = 0; i < numTriangles; i++) {
        polysArray[i * 4] = 3;
        polysArray[i * 4 + 1] = flatFaces[i * 3];
        polysArray[i * 4 + 2] = flatFaces[i * 3 + 1];
        polysArray[i * 4 + 3] = flatFaces[i * 3 + 2];
      }

      try {
        await geometryLoader.createAndCacheGeometry(geometryId, {
          type: coreEnums.GeometryType.SURFACE,
          geometryData: {
            id: geometryId,
            points,
            polys: polysArray,
            frameOfReferenceUID,
            segmentIndex,
          },
        });
      } catch (cacheErr: any) {
        console.warn(`[SurfaceLoader] Geometry cache failed for ${name}:`, cacheErr?.message);
      }

      const polyData = vtkPolyData.newInstance();
      polyData.getPoints().setData(points, 3);
      polyData.getPolys().setData(polysArray);

      const mapper = vtkMapper.newInstance();
      mapper.setInputData(polyData);

      const actor = vtkActor.newInstance();
      actor.setMapper(mapper);

      const prop = actor.getProperty();
      prop.setColor(mesh.color[0] / 255, mesh.color[1] / 255, mesh.color[2] / 255);
      prop.setOpacity(1.0);

      const uid = `nerve-surface-${name}`;
      actorEntries.push({ uid, actor });
      actorUIDs.push(uid);

      console.log(`[SurfaceLoader] Created actor: ${name} (${mesh.vertexCount}v, ${mesh.faceCount}f)`);
      segmentIndex++;
    }

    if (actorEntries.length > 0 && viewport.addActors) {
      viewport.addActors(actorEntries);
      addedActorUIDs.set(studyUID, actorUIDs);

      try {
        const actors = viewport.getActors?.() || [];
        for (const actorEntry of actors) {
          const actor = actorEntry.actor || actorEntry;
          if (actor?.getClassName?.() === 'vtkVolume') {
            applyBonePresetToActor(actor);

            const ofun = actor.getProperty().getScalarOpacity(0);
            ofun.removeAllPoints();
            ofun.addPoint(-1024, 0.0);
            ofun.addPoint(300, 0.0);
            ofun.addPoint(400, 0.15);
            ofun.addPoint(3000, 0.15);
            ofun.addPoint(3071, 0.15);

            console.log('[SurfaceLoader] Bone preset + HU threshold applied');
            break;
          }
        }
        reset2DViewportRendering(servicesManager);
      } catch (presetErr: any) {
        console.warn('[SurfaceLoader] Could not apply bone preset:', presetErr?.message);
      }

      viewport.render();
    }

    return true;
  } catch (error) {
    console.error('[SurfaceLoader] Failed to load surfaces:', error);
    return false;
  }
}

export function setVolumeHUThreshold(
  servicesManager: any,
  minHU: number = 200,
  maxHU: number = 2000,
  opacity: number = 0.15,
  rampWidth: number = 100,
): boolean {
  try {
    const viewportId = findVolume3DViewportId(servicesManager);
    if (!viewportId) return false;

    const { cornerstoneViewportService } = servicesManager.services;
    const viewport = cornerstoneViewportService?.getCornerstoneViewport?.(viewportId);
    if (!viewport) return false;

    const actors = viewport.getActors?.() || [];
    let volumeActor: any = null;

    for (const actorEntry of actors) {
      const actor = actorEntry.actor || actorEntry;
      if (actor?.getClassName?.() === 'vtkVolume') {
        volumeActor = actor;
        break;
      }
    }

    if (!volumeActor) {
      console.warn('[SurfaceLoader] No VTK volume actor found');
      return false;
    }

    const ofun = volumeActor.getProperty().getScalarOpacity(0);
    if (!ofun) {
      console.warn('[SurfaceLoader] No scalar opacity function found');
      return false;
    }

    ofun.removeAllPoints();
    ofun.addPoint(-1024, 0.0);
    ofun.addPoint(minHU, 0.0);
    ofun.addPoint(minHU + rampWidth, opacity);
    ofun.addPoint(maxHU, opacity);
    ofun.addPoint(3071, opacity);

    viewport.render();
    console.log(`[SurfaceLoader] HU threshold set: min=${minHU}, max=${maxHU}, opacity=${opacity}`);
    return true;
  } catch (e: any) {
    console.warn('[SurfaceLoader] Could not set HU threshold:', e?.message || e);
    return false;
  }
}

export async function removeSurfaceMeshes(
  studyUID: string,
  servicesManager: any
): Promise<void> {
  const viewportId = findVolume3DViewportId(servicesManager);
  if (!viewportId) return;

  const { cornerstoneViewportService } = servicesManager.services;
  const viewport = cornerstoneViewportService?.getCornerstoneViewport?.(viewportId);
  if (!viewport) return;

  const uids = addedActorUIDs.get(studyUID);
  if (uids && viewport.removeActors) {
    try {
      viewport.removeActors(uids);
      viewport.render();
      addedActorUIDs.delete(studyUID);
      console.log(`[SurfaceLoader] Removed ${uids.length} surface actors`);
    } catch (error) {
      console.error('[SurfaceLoader] Failed to remove surfaces:', error);
    }
  }
}

/**
 * Reset 2D viewport rendering after 3D bone preset leaks via shared VTK transfer functions.
 */
function reset2DViewportRendering(servicesManager: any): void {
  try {
    const { cornerstoneViewportService } = servicesManager.services;
    const allIds: string[] = cornerstoneViewportService?.getViewportIds?.() || [];

    for (const vpId of allIds) {
      const vpInfo = cornerstoneViewportService?.getViewportInfo?.(vpId);
      if (vpInfo?.viewportOptions?.viewportType === 'volume3d') continue;

      const vp = cornerstoneViewportService?.getCornerstoneViewport?.(vpId);
      if (!vp?.getActors) continue;

      const actors = vp.getActors();
      if (!actors || actors.length === 0) continue;

      const ctActorEntry = actors.find((a: any) =>
        (a.actor || a)?.getClassName?.() === 'vtkVolume'
      );
      if (!ctActorEntry) continue;

      const ctActor = ctActorEntry.actor || ctActorEntry;
      const prop = ctActor.getProperty();

      const ofun = prop.getScalarOpacity(0);
      ofun.removeAllPoints();
      ofun.addPoint(-3024, 1.0);
      ofun.addPoint(3071, 1.0);

      const cfun = prop.getRGBTransferFunction(0);
      cfun.removeAllPoints();
      cfun.addRGBPoint(-3024, 0, 0, 0);
      cfun.addRGBPoint(3071, 1, 1, 1);

      prop.setUseGradientOpacity(0, false);
      prop.setAmbient(1.0);
      prop.setDiffuse(0.0);
      prop.setSpecular(0.0);

      // W:350 / L:40 (head/neck soft tissue default)
      vp.setProperties({ voiRange: { lower: -135, upper: 215 } });
      vp.render();
    }
    console.log('[SurfaceLoader] 2D viewport rendering reset');
  } catch (e) {
    console.warn('[SurfaceLoader] 2D viewport reset failed:', e);
  }
}

async function reapplyOnLayoutChange(): Promise<void> {
  if (!currentServicesManager || !currentStudyUID) return;

  await new Promise(resolve => setTimeout(resolve, 500));
  console.log('[SurfaceLoader] Layout changed - reapplying meshes and labelmap');

  try {
    const reapplied = await reapplyLabelmapToViewports(currentStudyUID, currentServicesManager);
    if (reapplied) {
      console.log('[SurfaceLoader] Labelmap representation reapplied');
    }
  } catch (e) {
    console.warn('[SurfaceLoader] Labelmap reapply failed:', e);
  }

  const viewportId = findVolume3DViewportId(currentServicesManager);
  if (!viewportId) {
    console.log('[SurfaceLoader] No volume3d viewport in new layout');
    return;
  }

  try {
    const presetSet = setVolume3DPreset(currentServicesManager, 'CT-Bone');
    if (presetSet) {
      setVolumeHUThreshold(currentServicesManager, 300, 3000, 0.15, 100);
      reset2DViewportRendering(currentServicesManager);
    }
  } catch (e) {
    console.warn('[SurfaceLoader] Preset reapply failed:', e);
  }

  const cachedData = meshDataCache.get(currentStudyUID);
  if (cachedData) {
    try {
      await loadSurfaceMeshesFromCache(currentStudyUID, currentServicesManager, cachedData);

      // Restore 3D actor visibility from segment states
      const segStates = getSegmentStates(currentStudyUID);
      const cornerstone = (window as any).cornerstone || (window as any).cornerstoneCore;
      const re = cornerstone?.getRenderingEngine?.('OHIFCornerstoneRenderingEngine');
      const vp3d = re?.getViewports?.().find((v: any) => v.type === 'volume3d');
      if (vp3d) {
        for (const seg of segStates) {
          if (!seg.isVisible) {
            const actor = vp3d.getActors().find((a: any) => a.uid === `nerve-surface-${seg.label}`);
            if (actor) {
              actor.actor.setVisibility(false);
            }
          }
        }
        vp3d.render();
      }
    } catch (e) {
      console.warn('[SurfaceLoader] Mesh reapply failed:', e);
    }
  }
}

async function loadSurfaceMeshesFromCache(
  studyUID: string,
  servicesManager: any,
  meshData: SurfaceMeshResponse
): Promise<boolean> {
  const libs = getCornerstoneLibs();
  if (!libs) return false;

  const viewportId = findVolume3DViewportId(servicesManager);
  if (!viewportId) return false;

  const { cornerstoneViewportService } = servicesManager.services;
  const viewport = cornerstoneViewportService.getCornerstoneViewport?.(viewportId);
  if (!viewport) return false;

  const existingUIDs = addedActorUIDs.get(studyUID);
  if (existingUIDs && viewport.removeActors) {
    try {
      viewport.removeActors(existingUIDs);
      addedActorUIDs.delete(studyUID);
    } catch (e) {}
  }

  let vtkPolyData: any, vtkMapper: any, vtkActor: any;
  try {
    const polyDataMod = await import('@kitware/vtk.js/Common/DataModel/PolyData');
    const mapperMod = await import('@kitware/vtk.js/Rendering/Core/Mapper');
    const actorMod = await import('@kitware/vtk.js/Rendering/Core/Actor');
    vtkPolyData = polyDataMod.default || polyDataMod;
    vtkMapper = mapperMod.default || mapperMod;
    vtkActor = actorMod.default || actorMod;
  } catch (e) {
    console.error('[SurfaceLoader] Failed to import vtk.js:', e);
    return false;
  }

  const actorEntries: Array<{ uid: string; actor: any }> = [];
  const actorUIDs: string[] = [];

  for (const [name, mesh] of Object.entries(meshData.structures)) {
    const points = new Float32Array(mesh.vertices.flat());
    const numTriangles = mesh.faceCount;
    const polysArray = new Uint32Array(numTriangles * 4);
    const flatFaces = mesh.faces.flat();

    for (let i = 0; i < numTriangles; i++) {
      polysArray[i * 4] = 3;
      polysArray[i * 4 + 1] = flatFaces[i * 3];
      polysArray[i * 4 + 2] = flatFaces[i * 3 + 1];
      polysArray[i * 4 + 3] = flatFaces[i * 3 + 2];
    }

    const polyData = vtkPolyData.newInstance();
    polyData.getPoints().setData(points, 3);
    polyData.getPolys().setData(polysArray);

    const mapper = vtkMapper.newInstance();
    mapper.setInputData(polyData);

    const actor = vtkActor.newInstance();
    actor.setMapper(mapper);

    const prop = actor.getProperty();
    prop.setColor(mesh.color[0] / 255, mesh.color[1] / 255, mesh.color[2] / 255);
    prop.setOpacity(1.0);

    const uid = `nerve-surface-${name}`;
    actorEntries.push({ uid, actor });
    actorUIDs.push(uid);
  }

  if (actorEntries.length > 0 && viewport.addActors) {
    viewport.addActors(actorEntries);
    addedActorUIDs.set(studyUID, actorUIDs);
    viewport.render();
    console.log(`[SurfaceLoader] Reapplied ${actorEntries.length} cached surface actors`);
  }

  return true;
}

export function subscribeToLayoutChanges(servicesManager: any, studyUID: string): void {
  currentServicesManager = servicesManager;
  currentStudyUID = studyUID;

  if (typeof layoutChangeUnsubscribe === 'function') {
    layoutChangeUnsubscribe();
  }
  layoutChangeUnsubscribe = null;

  try {
    const { viewportGridService } = servicesManager.services;
    if (!viewportGridService?.subscribe) {
      console.warn('[SurfaceLoader] viewportGridService.subscribe not available');
      return;
    }

    const eventNames = [
      viewportGridService.EVENTS?.LAYOUT_CHANGED,
      viewportGridService.EVENTS?.GRID_STATE_CHANGED,
      'LAYOUT_CHANGED',
      'GRID_STATE_CHANGED',
    ].filter(Boolean);

    let subscribed = false;
    for (const eventName of eventNames) {
      try {
        const unsubscribe = viewportGridService.subscribe(eventName, () => {
          console.log('[SurfaceLoader] Layout change detected via', eventName);
          reapplyOnLayoutChange();
        });
        layoutChangeUnsubscribe = unsubscribe;
        console.log('[SurfaceLoader] Subscribed to layout changes via:', eventName);
        subscribed = true;
        break;
      } catch (subErr) {}
    }

    if (!subscribed) {
      console.warn('[SurfaceLoader] Could not subscribe to any layout event');
    }
  } catch (e) {
    console.warn('[SurfaceLoader] Failed to subscribe to layout changes:', e);
  }
}

export function unsubscribeFromLayoutChanges(): void {
  if (typeof layoutChangeUnsubscribe === 'function') {
    layoutChangeUnsubscribe();
  }
  layoutChangeUnsubscribe = null;
  currentServicesManager = null;
  currentStudyUID = null;
  meshDataCache.clear();
}

// ── Auto-Rotation ──────────────────────────────────────────────────────────

let rotationAnimationId: number | null = null;
let rotationDegreesPerFrame = 0.5;
let rotationOnStopCallback: (() => void) | null = null;
let rotationInteractorSub: any = null;

export function startAutoRotation(
  servicesManager: any,
  onStop?: () => void,
): void {
  // Stop any existing rotation first
  stopAutoRotation();

  rotationOnStopCallback = onStop || null;

  const cornerstone = (window as any).cornerstone || (window as any).cornerstoneCore;
  const re = cornerstone?.getRenderingEngine?.('OHIFCornerstoneRenderingEngine');
  const vp3d = re?.getViewports?.().find((v: any) => v.type === 'volume3d');
  if (!vp3d) {
    console.warn('[SurfaceLoader] No volume3d viewport for auto-rotation');
    return;
  }

  const renderer = vp3d.getRenderer?.();
  const camera = renderer?.getActiveCamera?.();
  if (!renderer || !camera) {
    console.warn('[SurfaceLoader] Could not get VTK renderer/camera');
    return;
  }

  // Register pointer listener on viewport canvas: stop rotation on mouse interaction
  try {
    const canvas = vp3d.canvas;
    if (canvas) {
      const handler = () => { stopAutoRotation(); };
      canvas.addEventListener('pointerdown', handler);
      rotationInteractorSub = { el: canvas, handler };
    }
  } catch (e) {
    console.warn('[SurfaceLoader] Could not register pointer listener:', e);
  }

  const animate = () => {
    if (!vp3d.getRenderer?.()) {
      stopAutoRotation();
      return;
    }
    camera.azimuth(rotationDegreesPerFrame);
    renderer.resetCameraClippingRange();
    vp3d.render();
    rotationAnimationId = requestAnimationFrame(animate);
  };

  rotationAnimationId = requestAnimationFrame(animate);
  console.log('[SurfaceLoader] Auto-rotation started, deg/frame:', rotationDegreesPerFrame);
}

export function stopAutoRotation(): void {
  if (rotationAnimationId !== null) {
    cancelAnimationFrame(rotationAnimationId);
    rotationAnimationId = null;
    console.log('[SurfaceLoader] Auto-rotation stopped');
  }

  if (rotationInteractorSub) {
    try {
      rotationInteractorSub.el.removeEventListener('pointerdown', rotationInteractorSub.handler);
    } catch (_) {}
    rotationInteractorSub = null;
  }

  if (rotationOnStopCallback) {
    const cb = rotationOnStopCallback;
    rotationOnStopCallback = null;
    cb();
  }
}

export function setRotationSpeed(degreesPerFrame: number): void {
  rotationDegreesPerFrame = degreesPerFrame;
}

export function isAutoRotating(): boolean {
  return rotationAnimationId !== null;
}

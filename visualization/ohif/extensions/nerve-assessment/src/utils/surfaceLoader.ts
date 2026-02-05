/**
 * Surface Loader for 3D mesh rendering.
 *
 * Fetches mesh data from backend REST API and renders it directly
 * in the volume3d viewport using VTK actors.
 *
 * IMPORTANT: Bypasses OHIF segmentation API entirely.
 * OHIF's addSegmentations() triggers SEGMENTATION_ADDED event handlers
 * that assume DICOM volume-based segmentations (imageIds required).
 * Standalone surface meshes from backend have no imageIds,
 * causing cascading crashes:
 *   addSegmentations() → SEGMENTATION_ADDED event
 *   → SegmentationService tries to read imageIds → undefined → crash
 *   → displaySetInstanceUID undefined → Error Boundary → mode exit
 *
 * Instead uses:
 * 1. geometryLoader.createAndCacheGeometry() — cache surface geometry
 * 2. Direct VTK actor creation from cached polydata
 * 3. viewport.addActors() — render in volume3d viewport
 * 4. VTK actor property for color/opacity (no segmentation color API)
 */

import { reapplyLabelmapToViewports } from './onDemandSegmentationManager';

// Volume3D viewport ID (must match hangingProtocol.ts)
export const VOLUME3D_VIEWPORT_ID = 'volume3d';

// Types for mesh data from backend
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

// Track added actor UIDs per study for cleanup
const addedActorUIDs = new Map<string, string[]>();

// Cache mesh data per study for reuse on layout changes
const meshDataCache = new Map<string, SurfaceMeshResponse>();

// Track layout change subscription
let layoutChangeUnsubscribe: (() => void) | null = null;
let currentServicesManager: any = null;
let currentStudyUID: string | null = null;

/**
 * Set volume rendering preset on the volume3d viewport.
 * Makes the CT volume visible as a 3D rendered body alongside surface actors.
 * Without this, the volume3d viewport only shows VTK surface actors.
 */
export function setVolume3DPreset(
  servicesManager: any,
  presetName: string = 'CT-Bone'
): boolean {
  try {
    const viewportId = findVolume3DViewportId(servicesManager);
    if (!viewportId) return false;

    const { cornerstoneViewportService } = servicesManager.services;
    const viewport = cornerstoneViewportService?.getCornerstoneViewport?.(viewportId);
    if (!viewport) return false;

    viewport.setProperties({ preset: presetName });
    viewport.render();
    console.log(`[SurfaceLoader] Volume3D preset set: ${presetName}`);
    return true;
  } catch (e: any) {
    console.warn('[SurfaceLoader] Could not set volume preset:', e?.message || e);
    return false;
  }
}

/**
 * Fetch surface mesh data from backend API.
 */
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

/**
 * Get cornerstone libraries from window globals.
 * OHIF v3.11 exposes these on window during build.
 */
function getCornerstoneLibs(): { cornerstone: any; cornerstoneTools: any } | null {
  const cornerstone = (window as any).cornerstone || (window as any).cornerstoneCore;
  const cornerstoneTools = (window as any).cornerstoneTools;

  if (!cornerstone || !cornerstoneTools) {
    console.error('[SurfaceLoader] Cornerstone libraries not found on window');
    return null;
  }

  return { cornerstone, cornerstoneTools };
}

/**
 * Find the volume3d viewport ID from servicesManager.
 * Falls back to the constant VOLUME3D_VIEWPORT_ID.
 */
function findVolume3DViewportId(servicesManager: any): string | null {
  try {
    const { cornerstoneViewportService } = servicesManager.services;
    const viewportIds = cornerstoneViewportService.getViewportIds?.() || [];

    for (const vpId of viewportIds) {
      const vpInfo = cornerstoneViewportService.getViewportInfo?.(vpId);
      const viewportOptions = vpInfo?.viewportOptions;
      if (viewportOptions?.viewportType === 'volume3d') {
        return vpId;
      }
    }

    // Fallback: check if the constant ID is available
    if (viewportIds.includes(VOLUME3D_VIEWPORT_ID)) {
      return VOLUME3D_VIEWPORT_ID;
    }
  } catch (e) {
    console.warn('[SurfaceLoader] Error finding volume3d viewport:', e);
  }

  return null;
}

/**
 * Load surface meshes as direct VTK actors in the volume3d viewport.
 *
 * Bypasses OHIF segmentation API to avoid DICOM-assumption crashes.
 *
 * Steps:
 * 1. Fetch mesh JSON from backend
 * 2. createAndCacheGeometry for each structure (cornerstone geometry cache)
 * 3. Get cached geometry → extract VTK polydata
 * 4. Create VTK mapper + actor, set color/opacity via actor.getProperty()
 * 5. viewport.addActors() to render in volume3d viewport
 */
export async function loadSurfaceMeshesToViewport(
  studyUID: string,
  servicesManager: any
): Promise<boolean> {
  console.log(`[SurfaceLoader] Loading surfaces for study: ${studyUID}`);

  // Get cornerstone libraries
  const libs = getCornerstoneLibs();
  if (!libs) return false;

  const { cornerstone } = libs;

  // Find volume3d viewport
  const viewportId = findVolume3DViewportId(servicesManager);
  if (!viewportId) {
    console.error('[SurfaceLoader] No volume3d viewport found. Check hanging protocol.');
    return false;
  }

  // Get the actual Cornerstone3D viewport object
  const { cornerstoneViewportService } = servicesManager.services;
  const viewport = cornerstoneViewportService.getCornerstoneViewport?.(viewportId);
  if (!viewport) {
    console.error('[SurfaceLoader] Could not get volume3d viewport object');
    return false;
  }
  console.log(`[SurfaceLoader] Using viewport: ${viewportId}`);

  // Fetch mesh data from backend (or use cache)
  let meshData = meshDataCache.get(studyUID);
  if (!meshData) {
    meshData = await fetchSurfaceMeshes(studyUID);
    if (meshData) {
      meshDataCache.set(studyUID, meshData);
      console.log('[SurfaceLoader] Mesh data cached for study:', studyUID);
    }
  } else {
    console.log('[SurfaceLoader] Using cached mesh data');
  }

  if (!meshData || meshData.structureCount === 0) {
    console.log('[SurfaceLoader] No mesh data available');
    return false;
  }

  const { geometryLoader, cache, Enums: coreEnums } = cornerstone;

  // Verify API availability
  if (!geometryLoader?.createAndCacheGeometry) {
    console.error('[SurfaceLoader] geometryLoader.createAndCacheGeometry not available');
    console.log('[SurfaceLoader] cornerstone keys:', Object.keys(cornerstone));
    return false;
  }

  // Get frameOfReferenceUID from CT volume viewport (required by cornerstone3D SurfaceData)
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
    } catch (_) { /* skip viewport */ }
  }

  if (!frameOfReferenceUID) {
    console.error('[SurfaceLoader] Could not get frameOfReferenceUID from CT volume');
    return false;
  }
  console.log('[SurfaceLoader] frameOfReferenceUID:', frameOfReferenceUID);

  // Import vtk.js for direct polydata + mapper + actor creation.
  // vtk.js is available as a dependency of @cornerstonejs/core in OHIF's webpack bundle.
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

      // Flatten vertices: [[x,y,z], ...] → Float32Array
      const points = new Float32Array(mesh.vertices.flat());

      // Convert faces to VTK connectivity: [nPts, p0, p1, p2, ...]
      const flatFaces = mesh.faces.flat();
      const numTriangles = mesh.faceCount;
      const polysArray = new Uint32Array(numTriangles * 4);
      for (let i = 0; i < numTriangles; i++) {
        polysArray[i * 4] = 3;
        polysArray[i * 4 + 1] = flatFaces[i * 3];
        polysArray[i * 4 + 2] = flatFaces[i * 3 + 1];
        polysArray[i * 4 + 3] = flatFaces[i * 3 + 2];
      }

      // Cache geometry in cornerstone3D (for potential contour conversion use)
      try {
        await geometryLoader.createAndCacheGeometry(geometryId, {
          type: coreEnums.GeometryType.SURFACE,
          geometryData: {
            id: geometryId,
            points: points,
            polys: polysArray,
            frameOfReferenceUID: frameOfReferenceUID,
            segmentIndex: segmentIndex,
          },
        });
      } catch (cacheErr: any) {
        console.warn(`[SurfaceLoader] Geometry cache failed for ${name}:`, cacheErr?.message);
        // Non-fatal — continue with direct VTK actor creation
      }

      // Create VTK polydata directly (bypass geometry cache retrieval)
      const polyData = vtkPolyData.newInstance();
      polyData.getPoints().setData(points, 3);
      polyData.getPolys().setData(polysArray);

      // Create mapper + actor
      const mapper = vtkMapper.newInstance();
      mapper.setInputData(polyData);

      const actor = vtkActor.newInstance();
      actor.setMapper(mapper);

      // Set color (VTK uses 0.0–1.0 range)
      const prop = actor.getProperty();
      prop.setColor(
        mesh.color[0] / 255,
        mesh.color[1] / 255,
        mesh.color[2] / 255
      );
      prop.setOpacity(1.0);

      const uid = `nerve-surface-${name}`;
      actorEntries.push({ uid, actor });
      actorUIDs.push(uid);

      console.log(`[SurfaceLoader] Created actor: ${name} (${mesh.vertexCount}v, ${mesh.faceCount}f)`);
      segmentIndex++;
    }

    console.log(`[SurfaceLoader] Created ${actorEntries.length} actors`);

    // Step 2: Add all actors to the volume3d viewport
    if (actorEntries.length > 0 && viewport.addActors) {
      viewport.addActors(actorEntries);
      addedActorUIDs.set(studyUID, actorUIDs);
      console.log(`[SurfaceLoader] Added ${actorEntries.length} surface actors to viewport: ${viewportId}`);

      // Step 3: Set CT volume rendering preset + HU threshold
      try {
        viewport.setProperties({ preset: 'CT-Bone' });
        console.log('[SurfaceLoader] Volume rendering preset set: CT-Bone');

        // Apply HU threshold immediately after preset to hide soft tissue (red)
        const actors = viewport.getActors?.() || [];
        for (const actorEntry of actors) {
          const actor = actorEntry.actor || actorEntry;
          if (actor?.getClassName?.() === 'vtkVolume') {
            const property = actor.getProperty();
            const ofun = property.getScalarOpacity(0);
            if (ofun) {
              ofun.removeAllPoints();
              ofun.addPoint(-1024, 0.0);
              ofun.addPoint(300, 0.0);
              ofun.addPoint(400, 0.15);
              ofun.addPoint(3000, 0.15);
              ofun.addPoint(3071, 0.15);
              console.log('[SurfaceLoader] HU threshold reapplied after preset');
            }
            break;
          }
        }
      } catch (presetErr: any) {
        console.warn('[SurfaceLoader] Could not set volume preset:', presetErr?.message);
      }

      viewport.render();
    }

    return true;
  } catch (error) {
    console.error('[SurfaceLoader] Failed to load surfaces:', error);
    return false;
  }
}

/**
 * Set custom HU threshold on volume3D viewport.
 * Modifies the VTK volume actor's opacity transfer function to control
 * which HU range is visible in 3D rendering.
 *
 * @param servicesManager - OHIF services manager
 * @param minHU - Minimum HU value to start showing (below = transparent)
 * @param maxHU - Maximum HU value (above = fully opaque at given opacity)
 * @param opacity - Maximum opacity (0.0 - 1.0)
 * @param rampWidth - HU width for opacity ramp-up (smooth transition)
 */
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

    // Get the VTK volume actor (the CT volume, not surface mesh actors)
    const actors = viewport.getActors?.() || [];
    let volumeActor: any = null;

    for (const actorEntry of actors) {
      // Volume actors have a volumeActor or are not our surface mesh actors
      const actor = actorEntry.actor || actorEntry;
      if (actor?.getClassName?.() === 'vtkVolume') {
        volumeActor = actor;
        break;
      }
    }

    if (!volumeActor) {
      console.warn('[SurfaceLoader] No VTK volume actor found in viewport');
      return false;
    }

    // Get the scalar opacity function (vtkPiecewiseFunction)
    const property = volumeActor.getProperty();
    const ofun = property.getScalarOpacity(0);

    if (!ofun) {
      console.warn('[SurfaceLoader] No scalar opacity function found');
      return false;
    }

    // Clear existing opacity points and set new ones
    ofun.removeAllPoints();

    // Build piecewise opacity function:
    // HU < minHU: transparent (0)
    // minHU to minHU+rampWidth: linear ramp from 0 to opacity
    // minHU+rampWidth to maxHU: constant opacity
    // HU > maxHU: constant opacity (bones are always high HU)
    ofun.addPoint(-1024, 0.0);               // air: transparent
    ofun.addPoint(minHU, 0.0);               // below threshold: transparent
    ofun.addPoint(minHU + rampWidth, opacity); // ramp up
    ofun.addPoint(maxHU, opacity);             // plateau
    ofun.addPoint(3071, opacity);              // max HU: keep visible

    viewport.render();
    console.log(`[SurfaceLoader] HU threshold set: min=${minHU}, max=${maxHU}, opacity=${opacity}, ramp=${rampWidth}`);
    return true;
  } catch (e: any) {
    console.warn('[SurfaceLoader] Could not set HU threshold:', e?.message || e);
    return false;
  }
}

/**
 * Remove surface actors from viewport.
 */
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
 * Reapply surface meshes, preset, and 2D labelmap when layout changes.
 * Uses cached mesh data to avoid re-fetching from backend.
 */
async function reapplyOnLayoutChange(): Promise<void> {
  if (!currentServicesManager || !currentStudyUID) {
    return;
  }

  // Wait a bit for viewports to initialize
  await new Promise(resolve => setTimeout(resolve, 500));

  console.log('[SurfaceLoader] Layout changed - reapplying meshes and labelmap');

  // Reapply 2D labelmap representation to new viewports (no re-fetch, no re-create)
  try {
    const reapplied = await reapplyLabelmapToViewports(currentStudyUID, currentServicesManager);
    if (reapplied) {
      console.log('[SurfaceLoader] ✓ Labelmap representation reapplied on layout change');
    } else {
      console.log('[SurfaceLoader] Labelmap reapply skipped (not loaded yet)');
    }
  } catch (e) {
    console.warn('[SurfaceLoader] Labelmap reapply failed:', e);
  }

  // Reapply 3D preset and surface meshes
  const viewportId = findVolume3DViewportId(currentServicesManager);
  if (!viewportId) {
    console.log('[SurfaceLoader] No volume3d viewport in new layout');
    return;
  }

  // Reapply preset and HU threshold
  try {
    const presetSet = setVolume3DPreset(currentServicesManager, 'CT-Bone');
    if (presetSet) {
      setVolumeHUThreshold(currentServicesManager, 300, 3000, 0.15, 100);
    }
  } catch (e) {
    console.warn('[SurfaceLoader] Preset reapply failed:', e);
  }

  // Reapply surface meshes from cache
  const cachedData = meshDataCache.get(currentStudyUID);
  if (cachedData) {
    try {
      await loadSurfaceMeshesFromCache(currentStudyUID, currentServicesManager, cachedData);
    } catch (e) {
      console.warn('[SurfaceLoader] Mesh reapply failed:', e);
    }
  } else {
    console.log('[SurfaceLoader] No cached mesh data for 3D reapply');
  }
}

/**
 * Load surface meshes from cached data (no network fetch).
 */
async function loadSurfaceMeshesFromCache(
  studyUID: string,
  servicesManager: any,
  meshData: SurfaceMeshResponse
): Promise<boolean> {
  const libs = getCornerstoneLibs();
  if (!libs) return false;

  const { cornerstone } = libs;
  const viewportId = findVolume3DViewportId(servicesManager);
  if (!viewportId) return false;

  const { cornerstoneViewportService } = servicesManager.services;
  const viewport = cornerstoneViewportService.getCornerstoneViewport?.(viewportId);
  if (!viewport) return false;

  // Remove existing actors first to prevent duplicates
  const existingUIDs = addedActorUIDs.get(studyUID);
  if (existingUIDs && viewport.removeActors) {
    try {
      viewport.removeActors(existingUIDs);
      addedActorUIDs.delete(studyUID);
      console.log('[SurfaceLoader] Removed existing actors before reapply');
    } catch (e) {
      // Ignore - actors may not exist in new viewport
    }
  }

  // Import VTK
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

/**
 * Subscribe to layout changes to reapply meshes when viewport structure changes.
 */
export function subscribeToLayoutChanges(servicesManager: any, studyUID: string): void {
  currentServicesManager = servicesManager;
  currentStudyUID = studyUID;

  // Unsubscribe previous listener if any
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

    // Log available events for debugging
    console.log('[SurfaceLoader] Available EVENTS:', viewportGridService.EVENTS);

    // Try known event names - LAYOUT_CHANGED preferred (only fires on actual layout change)
    // GRID_STATE_CHANGED fires too often (viewport clicks, active viewport changes)
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
      } catch (subErr) {
        // Try next event name
      }
    }

    if (!subscribed) {
      console.warn('[SurfaceLoader] Could not subscribe to any layout event');
    }
  } catch (e) {
    console.warn('[SurfaceLoader] Failed to subscribe to layout changes:', e);
  }
}

/**
 * Cleanup layout change subscription.
 */
export function unsubscribeFromLayoutChanges(): void {
  if (typeof layoutChangeUnsubscribe === 'function') {
    layoutChangeUnsubscribe();
  }
  layoutChangeUnsubscribe = null;
  currentServicesManager = null;
  currentStudyUID = null;
  meshDataCache.clear();
  console.log('[SurfaceLoader] Unsubscribed from layout changes');
}

/**
 * On-Demand Segmentation Manager
 *
 * Manages LABELMAP → CONTOUR conversion on-demand to avoid browser memory issues.
 * Instead of converting all 22+ segments at once (which crashes the browser),
 * only visible segments are converted to CONTOUR/Surface.
 *
 * Key Strategy:
 * - Load entire labelmap as LABELMAP representation (cheap, just voxel data)
 * - Track which segments are visible
 * - Convert visible segments to CONTOUR on-demand using polySeg
 * - Clean up CONTOUR data when segment is hidden
 *
 * Memory Usage:
 * - Full labelmap: ~75MB (uint8, 512x512x300)
 * - Each CONTOUR segment: ~50-100MB (surface mesh)
 * - 22 segments at once: ~1.5GB+ (CRASH!)
 * - 3-5 visible segments: ~300-500MB (OK)
 */

import type { NiftiData, LabelConfig } from './niftiLabelmapLoader';

// Segment state tracking
interface SegmentState {
  labelIndex: number;
  label: string;
  color: number[];
  isVisible: boolean;
  hasContour: boolean;  // Has CONTOUR representation been computed
}

// Manager state
interface ManagerState {
  studyInstanceUID: string;
  segmentationId: string | null;
  volumeId: string | null;
  segments: Map<number, SegmentState>;  // labelIndex -> state
  isLoaded: boolean;
}

// Global state (singleton per study)
const managerStates = new Map<string, ManagerState>();

/**
 * Initialize manager for a study
 */
export function initializeManager(studyInstanceUID: string): ManagerState {
  if (managerStates.has(studyInstanceUID)) {
    return managerStates.get(studyInstanceUID)!;
  }

  const state: ManagerState = {
    studyInstanceUID,
    segmentationId: null,
    volumeId: null,
    segments: new Map(),
    isLoaded: false,
  };

  managerStates.set(studyInstanceUID, state);
  console.log(`[OnDemandManager] Initialized for study: ${studyInstanceUID.slice(0, 20)}...`);

  return state;
}

/**
 * Get manager state for a study
 */
export function getManagerState(studyInstanceUID: string): ManagerState | null {
  return managerStates.get(studyInstanceUID) || null;
}

/**
 * Get Cornerstone3D core from global (OHIF's instance)
 * IMPORTANT: Must use global instance to share state with OHIF
 */
function getCornerstoneCore(): any {
  const cs = (window as any).cornerstone || (window as any).cornerstoneCore;
  if (!cs) {
    throw new Error('Cornerstone core not found on window. OHIF may not be fully loaded.');
  }
  return cs;
}

/**
 * Get Cornerstone Tools from global (OHIF's instance)
 * IMPORTANT: Must use global instance to share segmentation state
 */
function getCornerstoneTools(): any {
  const csTools = (window as any).cornerstoneTools;
  if (!csTools) {
    throw new Error('Cornerstone Tools not found on window. OHIF may not be fully loaded.');
  }
  return csTools;
}

/**
 * Get all MPR (volume) viewport IDs dynamically from servicesManager.
 * OHIF may assign different IDs depending on entry path (upload vs study list).
 * Filters out volume3d viewports.
 */
function getMPRViewportIds(servicesManager: any): string[] {
  try {
    const { cornerstoneViewportService } = servicesManager.services;
    const allIds: string[] = cornerstoneViewportService?.getViewportIds?.() || [];
    const mprIds: string[] = [];

    for (const vpId of allIds) {
      const vpInfo = cornerstoneViewportService?.getViewportInfo?.(vpId);
      const vpOptions = vpInfo?.viewportOptions;
      // Include volume viewports, exclude volume3d
      if (vpOptions?.viewportType === 'volume' || vpOptions?.viewportType === 'orthographic') {
        mprIds.push(vpId);
      } else if (!vpOptions?.viewportType || vpOptions?.viewportType === 'stack') {
        // Stack viewports might also display segmentations in some modes
        mprIds.push(vpId);
      }
      // Skip volume3d viewports
    }

    if (mprIds.length === 0) {
      // Fallback: return all viewport IDs except known 3D ones
      return allIds.filter(id => id !== 'volume3d');
    }

    console.log('[OnDemandManager] MPR viewport IDs:', mprIds);
    return mprIds;
  } catch (e) {
    console.warn('[OnDemandManager] Failed to get viewport IDs:', e);
    return [];
  }
}

/**
 * Get polySeg from global or OHIF extension
 * polySeg may be registered as window.cornerstonePolySeg or via extension
 */
function getPolySeg(servicesManager?: any): any {
  // Try global first
  const polySeg = (window as any).cornerstonePolySeg ||
                  (window as any).polySeg ||
                  (window as any)['@cornerstonejs/polymorphic-segmentation'];

  if (polySeg) {
    return polySeg;
  }

  // Try via OHIF extension manager
  if (servicesManager?.services?.extensionManager) {
    try {
      const extManager = servicesManager.services.extensionManager;
      // polySeg might be exposed via cornerstone extension
      const csExtension = extManager.getExtension?.('@ohif/extension-cornerstone');
      if (csExtension?.polySeg) {
        return csExtension.polySeg;
      }
    } catch (e) {
      // Extension not available
    }
  }

  return null;
}

/**
 * Create a Cornerstone3D volume from NIfTI data
 */
export async function createLabelmapVolume(
  niftiData: NiftiData,
  labelConfig: LabelConfig,
  studyInstanceUID: string,
  servicesManager: any
): Promise<string> {
  const state = initializeManager(studyInstanceUID);

  // Generate unique volume ID
  const volumeId = `labelmap_${studyInstanceUID}_${Date.now()}`;

  try {
    // Use global Cornerstone instance (shared with OHIF)
    const cornerstone = getCornerstoneCore();
    const { volumeLoader, cache } = cornerstone;

    console.log('[OnDemandManager] Creating labelmap volume:', {
      volumeId,
      dimensions: niftiData.dimensions,
      spacing: niftiData.spacing,
      dataLength: niftiData.typedArray.length,
    });

    // Create volume metadata
    const volumeMetadata = {
      BitsAllocated: 8,
      BitsStored: 8,
      SamplesPerPixel: 1,
      HighBit: 7,
      PixelRepresentation: 0,
      PhotometricInterpretation: 'MONOCHROME2',
      Columns: niftiData.dimensions[0],
      Rows: niftiData.dimensions[1],
      NumberOfFrames: niftiData.dimensions[2],
      ImageOrientationPatient: niftiData.direction.slice(0, 6),
      ImagePositionPatient: niftiData.origin,
      PixelSpacing: [niftiData.spacing[0], niftiData.spacing[1]],
      SliceThickness: niftiData.spacing[2],
    };

    // Register the volume with the cache
    // Method depends on Cornerstone3D version
    if (volumeLoader?.createLocalVolume) {
      // Cornerstone3D v3.32.5 API: createLocalVolume(volumeId, options)
      const volume = await volumeLoader.createLocalVolume(volumeId, {
        scalarData: niftiData.typedArray,
        metadata: volumeMetadata,
        dimensions: niftiData.dimensions,
        spacing: niftiData.spacing,
        origin: niftiData.origin,
        direction: niftiData.direction,
      });
      console.log('[OnDemandManager] Volume created via createLocalVolume');
    } else {
      // Fallback: Create volume and add to cache manually
      const volume = {
        volumeId,
        dimensions: niftiData.dimensions,
        spacing: niftiData.spacing,
        origin: niftiData.origin,
        direction: niftiData.direction,
        metadata: volumeMetadata,
        scalarData: niftiData.typedArray,
        sizeInBytes: niftiData.typedArray.byteLength,
      };

      if (cache?.putVolumeLoadObject) {
        cache.putVolumeLoadObject(volumeId, { promise: Promise.resolve(volume) });
      } else if (cache?.setVolume) {
        cache.setVolume(volumeId, volume);
      }
      console.log('[OnDemandManager] Volume added to cache manually');
    }

    // Initialize segment states from label config
    for (const segment of labelConfig.segments) {
      state.segments.set(segment.segmentIndex, {
        labelIndex: segment.segmentIndex,
        label: segment.label,
        color: segment.color,
        isVisible: false,  // Start hidden, user toggles on
        hasContour: false,
      });
    }

    state.volumeId = volumeId;
    state.isLoaded = true;

    console.log(`[OnDemandManager] Labelmap ready with ${state.segments.size} segments`);

    return volumeId;
  } catch (error) {
    console.error('[OnDemandManager] Failed to create labelmap volume:', error);
    throw error;
  }
}

/**
 * Add labelmap as a segmentation to the segmentation service
 */
export async function addLabelmapSegmentation(
  volumeId: string,
  labelConfig: LabelConfig,
  studyInstanceUID: string,
  servicesManager: any
): Promise<string> {
  const state = getManagerState(studyInstanceUID);
  if (!state) {
    throw new Error('Manager not initialized');
  }

  const { segmentationService } = servicesManager.services;
  if (!segmentationService) {
    throw new Error('SegmentationService not available');
  }

  // Generate segmentation ID
  const segmentationId = `seg_labelmap_${studyInstanceUID.slice(0, 8)}`;

  try {
    // Use global Cornerstone Tools instance (shared with OHIF)
    const csTools = getCornerstoneTools();
    const { segmentation } = csTools;

    // Build segments metadata for PanelSegmentation
    // Without this, PanelSegmentation crashes with "Cannot read properties of undefined (reading '0')"
    const segmentsMetadata: Record<number, { segmentIndex: number; label: string; active: boolean; color?: number[] }> = {};
    for (const segment of labelConfig.segments) {
      segmentsMetadata[segment.segmentIndex] = {
        segmentIndex: segment.segmentIndex,
        label: segment.label,
        active: false,
        color: segment.color,
      };
    }

    // Create segmentation input with segments metadata
    const segmentationInput = {
      segmentationId,
      representation: {
        type: 'Labelmap',
        data: {
          volumeId,
        },
      },
      segments: segmentsMetadata,
    };

    // Add segmentation to state (Cornerstone3D v3.32.5 API: addSegmentations - plural, array input)
    if (segmentation?.addSegmentations) {
      segmentation.addSegmentations([segmentationInput]);
      console.log('[OnDemandManager] Segmentation added via addSegmentations()');
      // Inject segment metadata directly into state (PanelSegmentation needs this)
      const segObj = segmentation.state.getSegmentation(segmentationId);
      if (segObj && labelConfig?.segments) {
        for (const seg of labelConfig.segments) {
          segObj.segments[seg.segmentIndex] = {
            segmentIndex: seg.segmentIndex,
            label: seg.label,
            active: false,
            locked: false,
            cachedStats: {},
          };
        }
        console.log('[OnDemandManager] Injected ' + labelConfig.segments.length + ' segment metadata entries');
      }

      // Colors are set AFTER representation registration (below)
      // Setting colors here would be overwritten by addLabelmapRepresentationToViewport
      // and segmentationService.addSegmentation

      // Verify registration
      const allSegs = segmentation.state.getSegmentations() || [];
      console.log('[OnDemandManager] Registered segmentations after add:', allSegs.map((s: any) => s.segmentationId));

      if (!allSegs.find((s: any) => s.segmentationId === segmentationId)) {
        console.error('[OnDemandManager] WARNING: Segmentation was not found after registration!');
      }
    } else {
      console.error('[OnDemandManager] segmentation.addSegmentations not available');
    }

    // Add Labelmap representation to all MPR viewports
    // This is required for polySeg to convert Labelmap → Contour
    const viewportIds = getMPRViewportIds(servicesManager);

    for (const viewportId of viewportIds) {
      try {
        // Try the type-specific function first (Cornerstone3D v2+)
        if (segmentation?.addLabelmapRepresentationToViewport) {
          await segmentation.addLabelmapRepresentationToViewport(viewportId, [
            { segmentationId },
          ]);
          console.log(`[OnDemandManager] Labelmap representation added to ${viewportId} (v2 API)`);
        }
        // Fallback to general function
        else if (segmentation?.addSegmentationRepresentations) {
          await segmentation.addSegmentationRepresentations(viewportId, [
            {
              segmentationId,
              type: csTools.Enums?.SegmentationRepresentations?.Labelmap || 'Labelmap',
            },
          ]);
          console.log(`[OnDemandManager] Labelmap representation added to ${viewportId}`);
        }
      } catch (e: any) {
        console.warn(`[OnDemandManager] Failed to add representation to ${viewportId}:`, e?.message);
      }
    }

    // Also add via OHIF service if available
    if (segmentationService.addSegmentation) {
      try {
        await segmentationService.addSegmentation({
          segmentationId,
          label: 'NIfTI Labelmap',
          segments: labelConfig.segments.map(seg => ({
            segmentIndex: seg.segmentIndex,
            label: seg.label,
            color: seg.color,
            isVisible: false,  // Start hidden
            isLocked: false,
          })),
          type: 'LABELMAP',
          representationData: {
            Labelmap: {
              volumeId,
            },
          },
        });
        console.log('[OnDemandManager] Segmentation added to OHIF service');
      } catch (e: any) {
        console.warn('[OnDemandManager] OHIF addSegmentation failed:', e?.message);
      }
    }

    state.segmentationId = segmentationId;

    // Set segment colors using a custom Color LUT.
    // Individual setSegmentIndexColor calls get overwritten by OHIF's async processing.
    // Replacing the entire LUT via addColorLUT + setColorLUT is reliable.
    // API: addColorLUT(colorLUT[], lutIndex), setColorLUT(viewportId, segmentationId, lutIndex)
    const csToolsForColor = getCornerstoneTools();
    const colorApi = csToolsForColor.segmentation?.config?.color;
    if (colorApi?.addColorLUT && colorApi?.setColorLUT) {
      const CUSTOM_LUT_INDEX = 2; // avoid 0,1 which are OHIF defaults
      const maxIndex = Math.max(...labelConfig.segments.map(s => s.segmentIndex));
      const colorLUT: [number, number, number, number][] = [];
      // Index 0 = background (transparent)
      colorLUT[0] = [0, 0, 0, 0];
      for (let i = 1; i <= maxIndex; i++) {
        colorLUT[i] = [128, 128, 128, 150]; // default gray for unmapped
      }
      for (const seg of labelConfig.segments) {
        const rgba: [number, number, number, number] = seg.color.length >= 4
          ? [seg.color[0], seg.color[1], seg.color[2], seg.color[3]]
          : [seg.color[0], seg.color[1], seg.color[2], 255];
        colorLUT[seg.segmentIndex] = rgba;
      }

      try {
        colorApi.addColorLUT(colorLUT, CUSTOM_LUT_INDEX);
        const vpIds = getMPRViewportIds(servicesManager);
        for (const vpId of vpIds) {
          try {
            colorApi.setColorLUT(vpId, segmentationId, CUSTOM_LUT_INDEX);
          } catch (_) {}
        }
        console.log(`[OnDemandManager] Custom color LUT applied (${labelConfig.segments.length} segments, LUT index ${CUSTOM_LUT_INDEX})`);
      } catch (e: any) {
        console.warn('[OnDemandManager] Color LUT failed:', e?.message);
      }
    } else {
      console.warn('[OnDemandManager] addColorLUT/setColorLUT not available');
    }

    console.log(`[OnDemandManager] Segmentation registered: ${segmentationId}`);

    return segmentationId;
  } catch (error) {
    console.error('[OnDemandManager] Failed to add segmentation:', error);
    throw error;
  }
}

/**
 * Convert a specific segment to CONTOUR representation (on-demand)
 */
export async function convertSegmentToContour(
  segmentIndex: number,
  studyInstanceUID: string,
  viewportId: string,
  servicesManager: any
): Promise<boolean> {
  const state = getManagerState(studyInstanceUID);
  if (!state || !state.segmentationId) {
    console.warn('[OnDemandManager] Manager or segmentation not ready');
    return false;
  }

  const segmentState = state.segments.get(segmentIndex);
  if (!segmentState) {
    console.warn(`[OnDemandManager] Segment ${segmentIndex} not found`);
    return false;
  }

  if (segmentState.hasContour) {
    console.log(`[OnDemandManager] Segment ${segmentIndex} already has CONTOUR`);
    return true;
  }

  try {
    console.log(`[OnDemandManager] Converting segment ${segmentIndex} (${segmentState.label}) to CONTOUR...`);

    // Use global instances (shared with OHIF)
    const csTools = getCornerstoneTools();
    let polySeg = getPolySeg(servicesManager);

    if (!polySeg) {
      console.warn('[OnDemandManager] polySeg not found in global, trying dynamic import...');
      // Fallback: try dynamic import (may create separate instance - not ideal)
      try {
        polySeg = await import('@cornerstonejs/polymorphic-segmentation');
        console.warn('[OnDemandManager] Using dynamic import for polySeg - state may not be shared');
      } catch (e) {
        console.error('[OnDemandManager] polySeg not available:', e);
        return false;
      }
    }

    const { cornerstoneViewportService, segmentationService } = servicesManager.services;

    // Get viewport
    const viewport = cornerstoneViewportService?.getCornerstoneViewport?.(viewportId);
    if (!viewport) {
      console.warn('[OnDemandManager] Viewport not available');
      return false;
    }

    // Debug: Check segmentation state before polySeg call
    const { segmentation } = csTools;
    const allSegs = segmentation.state.getSegmentations();
    console.log('[OnDemandManager] Segmentations in state before polySeg:', allSegs.map((s: any) => s.segmentationId));
    console.log('[OnDemandManager] Looking for segmentationId:', state.segmentationId);

    if (!allSegs.find((s: any) => s.segmentationId === state.segmentationId)) {
      console.error('[OnDemandManager] Segmentation not found in CS Tools state! Cannot compute CONTOUR.');
      return false;
    }

    // Check if polySeg can compute
    if (polySeg.canComputeRequestedRepresentation) {
      const canCompute = polySeg.canComputeRequestedRepresentation(
        state.segmentationId,
        'Contour'
      );

      if (!canCompute) {
        console.warn('[OnDemandManager] polySeg cannot compute CONTOUR');
        return false;
      }
    }

    // Step 1: Compute surface from labelmap and cache geometry directly.
    // polySeg.computeSurfaceData internally calls createAndCacheSurfacesFromRaw
    // → geometryLoader.createAndCacheGeometry for each segment.
    // Returns { geometryIds: Map<segmentIndex, geometryId> }.
    //
    // We then manually set representationData.Surface on the segmentation object
    // so computeContourData finds it and takes the fast Surface→Contour path.
    //
    // Do NOT use computeAndAddRepresentation('Surface') — that registers the Surface
    // as an OHIF representation, which triggers updateSegmentationStats
    // → isValidVolume → MetadataProvider::Empty imageId (NIfTI has no DICOM imageIds).
    const computeAndAddRepresentation =
      csTools.utilities?.segmentation?.computeAndAddRepresentation;
    const toolsEnums = csTools.Enums;

    if (polySeg.computeSurfaceData) {
      const segObj = segmentation.state?.getSegmentation?.(state.segmentationId);
      const hasSurface = segObj?.representationData?.Surface?.geometryIds?.size > 0;

      if (!hasSurface) {
        console.log(`[OnDemandManager] Computing surface for segment ${segmentIndex}...`);
        const surfaceResult = await polySeg.computeSurfaceData(state.segmentationId, {
          segmentIndices: [segmentIndex],
          viewport,
        });

        // Manually set representationData.Surface (no representation registration)
        if (surfaceResult?.geometryIds && segObj) {
          if (!segObj.representationData.Surface) {
            segObj.representationData.Surface = { geometryIds: surfaceResult.geometryIds };
          } else {
            for (const [idx, geoId] of surfaceResult.geometryIds) {
              segObj.representationData.Surface.geometryIds.set(idx, geoId);
            }
          }
        }
        console.log(`[OnDemandManager] Surface cached for segment ${segmentIndex}`);
      }
    }

    // Step 2: Compute contour for each MPR viewport orientation.
    // With Surface in representationData + geometry cache, computeContourData
    // takes the fast Surface path (cached geometry → clipping → contour extraction).
    const mprViewportIds = getMPRViewportIds(servicesManager);
    let isFirstContour = true;

    for (const vpId of mprViewportIds) {
      const vp = cornerstoneViewportService?.getCornerstoneViewport?.(vpId);
      if (!vp) {
        console.warn(`[OnDemandManager] Viewport ${vpId} not ready, skipping`);
        continue;
      }

      try {
        if (isFirstContour && computeAndAddRepresentation && polySeg.computeContourData) {
          // First viewport: initialize Contour representation data via computeAndAddRepresentation
          await computeAndAddRepresentation(
            state.segmentationId,
            toolsEnums?.SegmentationRepresentations?.Contour || 'Contour',
            () => polySeg.computeContourData(state.segmentationId!, {
              viewport: vp,
              segmentIndices: [segmentIndex],
            }),
            () => undefined
          );
          isFirstContour = false;
        } else if (polySeg.computeContourData) {
          // Additional viewports: compute contour and merge annotationUIDsMap
          // (mirrors Phase 2 merge logic in cornerstone3D contourDisplay.ts)
          const newData = await polySeg.computeContourData(state.segmentationId, {
            viewport: vp,
            segmentIndices: [segmentIndex],
          });

          if (newData?.annotationUIDsMap) {
            const segObj = segmentation.state?.getSegmentation?.(state.segmentationId);
            const existing = segObj?.representationData?.Contour;
            if (existing?.annotationUIDsMap) {
              for (const [segIdx, uids] of newData.annotationUIDsMap) {
                const existingUids = existing.annotationUIDsMap.get(segIdx);
                if (existingUids) {
                  for (const uid of uids) existingUids.add(uid);
                } else {
                  existing.annotationUIDsMap.set(segIdx, uids);
                }
              }
            }
          }
        }
        console.log(`[OnDemandManager] Contour computed for ${vpId}`);
      } catch (e: any) {
        console.warn(`[OnDemandManager] Contour for ${vpId} failed:`, e?.message || e);
      }

      // Ensure contour representation is registered on this viewport
      try {
        if (segmentation.addContourRepresentationToViewport) {
          await segmentation.addContourRepresentationToViewport(vpId, [
            { segmentationId: state.segmentationId },
          ]);
        }
      } catch (_) {
        // Already exists - OK
      }
    }

    // Ensure contour color matches label config on all MPR viewports.
    // Cornerstone3D v3.32.5 API: setSegmentIndexColor(viewportId, segmentationId, segmentIndex, color)
    const setColor = segmentation.config?.color?.setSegmentIndexColor;
    if (setColor) {
      const rgba = segmentState.color.length >= 4
        ? [segmentState.color[0], segmentState.color[1], segmentState.color[2], segmentState.color[3]]
        : [segmentState.color[0], segmentState.color[1], segmentState.color[2], 255];
      for (const vpId of mprViewportIds) {
        try {
          setColor(vpId, state.segmentationId, segmentIndex, rgba);
        } catch (_) { /* viewport may not have representation yet */ }
      }
    }

    // Update state
    segmentState.hasContour = true;
    segmentState.isVisible = true;

    console.log(`[OnDemandManager] Segment ${segmentIndex} CONTOUR conversion complete`);

    // Trigger render on all MPR viewports
    for (const vpId of mprViewportIds) {
      const vp = cornerstoneViewportService?.getCornerstoneViewport?.(vpId);
      vp?.render?.();
    }

    return true;
  } catch (error) {
    console.error(`[OnDemandManager] CONTOUR conversion failed for segment ${segmentIndex}:`, error);
    return false;
  }
}

/**
 * Set segment visibility and trigger on-demand conversion if needed
 */
export async function setSegmentVisibility(
  segmentIndex: number,
  isVisible: boolean,
  studyInstanceUID: string,
  viewportId: string,
  servicesManager: any
): Promise<void> {
  const state = getManagerState(studyInstanceUID);
  if (!state || !state.segmentationId) {
    return;
  }

  const segmentState = state.segments.get(segmentIndex);
  if (!segmentState) {
    return;
  }

  console.log(`[OnDemandManager] Setting segment ${segmentIndex} visibility: ${isVisible}`);

  const { segmentationService } = servicesManager.services;

  // If showing and no CONTOUR yet, compute it
  if (isVisible && !segmentState.hasContour) {
    await convertSegmentToContour(segmentIndex, studyInstanceUID, viewportId, servicesManager);
  }

  // Update visibility in service
  if (segmentationService?.setSegmentVisibility) {
    try {
      segmentationService.setSegmentVisibility(
        state.segmentationId,
        segmentIndex,
        isVisible
      );
    } catch (e) {
      // API might differ
    }
  }

  segmentState.isVisible = isVisible;
}

/**
 * Toggle segment visibility
 */
export async function toggleSegmentVisibility(
  segmentIndex: number,
  studyInstanceUID: string,
  viewportId: string,
  servicesManager: any
): Promise<void> {
  const state = getManagerState(studyInstanceUID);
  if (!state) return;

  const segmentState = state.segments.get(segmentIndex);
  if (!segmentState) return;

  await setSegmentVisibility(
    segmentIndex,
    !segmentState.isVisible,
    studyInstanceUID,
    viewportId,
    servicesManager
  );
}

/**
 * Show only specific segments (hide all others)
 */
export async function showOnlySegments(
  segmentIndices: number[],
  studyInstanceUID: string,
  viewportId: string,
  servicesManager: any
): Promise<void> {
  const state = getManagerState(studyInstanceUID);
  if (!state) return;

  const indicesSet = new Set(segmentIndices);

  for (const [index, segmentState] of state.segments) {
    const shouldShow = indicesSet.has(index);
    if (segmentState.isVisible !== shouldShow) {
      await setSegmentVisibility(index, shouldShow, studyInstanceUID, viewportId, servicesManager);
    }
  }
}

/**
 * Get count of visible segments
 */
export function getVisibleSegmentCount(studyInstanceUID: string): number {
  const state = getManagerState(studyInstanceUID);
  if (!state) return 0;

  let count = 0;
  for (const segmentState of state.segments.values()) {
    if (segmentState.isVisible) count++;
  }
  return count;
}

/**
 * Get all segment states
 */
export function getSegmentStates(studyInstanceUID: string): SegmentState[] {
  const state = getManagerState(studyInstanceUID);
  if (!state) return [];

  return Array.from(state.segments.values());
}

/**
 * Reapply existing labelmap segmentation to current viewports after layout change.
 * Does NOT re-fetch NIfTI or recreate volume/segmentation.
 * Only adds representation + color LUT to new viewport IDs.
 */
export async function reapplyLabelmapToViewports(
  studyInstanceUID: string,
  servicesManager: any
): Promise<boolean> {
  const state = getManagerState(studyInstanceUID);
  if (!state || !state.segmentationId || !state.isLoaded) {
    console.log('[OnDemandManager] No existing labelmap to reapply');
    return false;
  }

  const segmentationId = state.segmentationId;
  console.log(`[OnDemandManager] Reapplying labelmap ${segmentationId} to new viewports`);

  try {
    const csTools = getCornerstoneTools();
    const { segmentation } = csTools;
    const viewportIds = getMPRViewportIds(servicesManager);

    if (viewportIds.length === 0) {
      console.log('[OnDemandManager] No MPR viewports found for reapply');
      return false;
    }

    // Add labelmap representation to each new viewport
    for (const viewportId of viewportIds) {
      try {
        if (segmentation?.addLabelmapRepresentationToViewport) {
          await segmentation.addLabelmapRepresentationToViewport(viewportId, [
            { segmentationId },
          ]);
          console.log(`[OnDemandManager] Labelmap reapplied to ${viewportId}`);
        } else if (segmentation?.addSegmentationRepresentations) {
          await segmentation.addSegmentationRepresentations(viewportId, [
            {
              segmentationId,
              type: csTools.Enums?.SegmentationRepresentations?.Labelmap || 'Labelmap',
            },
          ]);
          console.log(`[OnDemandManager] Labelmap reapplied to ${viewportId} (fallback API)`);
        }
      } catch (e: any) {
        // "already exists" is expected and OK
        if (e?.message?.includes('already') || e?.message?.includes('exist')) {
          console.log(`[OnDemandManager] Labelmap already on ${viewportId}`);
        } else {
          console.warn(`[OnDemandManager] Reapply to ${viewportId} failed:`, e?.message);
        }
      }
    }

    // Re-apply custom color LUT to new viewports
    const colorApi = csTools.segmentation?.config?.color;
    if (colorApi?.setColorLUT) {
      const CUSTOM_LUT_INDEX = 2;
      for (const vpId of viewportIds) {
        try {
          colorApi.setColorLUT(vpId, segmentationId, CUSTOM_LUT_INDEX);
        } catch (_) {}
      }
      console.log('[OnDemandManager] Color LUT reapplied to new viewports');
    }

    // Re-apply contour representation for segments that were already visible
    for (const [segIdx, segState] of state.segments) {
      if (segState.hasContour) {
        for (const vpId of viewportIds) {
          try {
            if (segmentation.addContourRepresentationToViewport) {
              await segmentation.addContourRepresentationToViewport(vpId, [
                { segmentationId },
              ]);
            }
          } catch (_) {
            // Already exists - OK
          }
        }
      }
    }

    // Trigger render on all viewports
    const { cornerstoneViewportService } = servicesManager.services;
    for (const vpId of viewportIds) {
      const vp = cornerstoneViewportService?.getCornerstoneViewport?.(vpId);
      vp?.render?.();
    }

    console.log('[OnDemandManager] Labelmap reapply complete');
    return true;
  } catch (error) {
    console.error('[OnDemandManager] Labelmap reapply failed:', error);
    return false;
  }
}

/**
 * Clean up manager for a study
 */
export function cleanupManager(studyInstanceUID: string): void {
  const state = managerStates.get(studyInstanceUID);
  if (!state) return;

  console.log(`[OnDemandManager] Cleaning up study: ${studyInstanceUID.slice(0, 20)}...`);

  // Clear segment states
  state.segments.clear();
  state.isLoaded = false;

  // Remove from global map
  managerStates.delete(studyInstanceUID);
}

/**
 * Subscribe to segment visibility changes from OHIF segmentation service
 */
export function subscribeToVisibilityChanges(
  servicesManager: any,
  studyInstanceUID: string,
  viewportId: string
): (() => void) | null {
  const { segmentationService } = servicesManager.services;
  if (!segmentationService?.subscribe) {
    console.warn('[OnDemandManager] Cannot subscribe - no subscribe method');
    return null;
  }

  const EVENTS = segmentationService.EVENTS || {};

  // Try different event names
  const visibilityEventNames = [
    'SEGMENT_VISIBILITY_CHANGED',
    'SEGMENTATION_VISIBILITY_CHANGED',
    EVENTS.SEGMENT_VISIBILITY_CHANGED,
    EVENTS.SEGMENTATION_VISIBILITY_CHANGED,
  ].filter(Boolean);

  const unsubscribes: (() => void)[] = [];

  for (const eventName of visibilityEventNames) {
    try {
      const unsub = segmentationService.subscribe(
        eventName,
        async (event: any) => {
          const { segmentIndex, isVisible, segmentationId } = event || {};

          if (segmentIndex !== undefined && isVisible !== undefined) {
            console.log(`[OnDemandManager] Visibility event: segment ${segmentIndex} -> ${isVisible}`);

            // Trigger on-demand conversion
            await setSegmentVisibility(
              segmentIndex,
              isVisible,
              studyInstanceUID,
              viewportId,
              servicesManager
            );
          }
        }
      );

      if (unsub) {
        unsubscribes.push(unsub);
        console.log(`[OnDemandManager] Subscribed to ${eventName}`);
      }
    } catch (e) {
      // Event might not exist
    }
  }

  // Return combined unsubscribe
  if (unsubscribes.length > 0) {
    return () => {
      unsubscribes.forEach(fn => fn());
    };
  }

  return null;
}

export default {
  initializeManager,
  getManagerState,
  createLabelmapVolume,
  addLabelmapSegmentation,
  reapplyLabelmapToViewports,
  convertSegmentToContour,
  setSegmentVisibility,
  toggleSegmentVisibility,
  showOnlySegments,
  getVisibleSegmentCount,
  getSegmentStates,
  cleanupManager,
  subscribeToVisibilityChanges,
};

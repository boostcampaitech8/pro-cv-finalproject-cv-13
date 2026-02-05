/**
 * Labelmap Integration for OHIF Nerve Assessment Mode
 *
 * Main entry point for loading NIfTI labelmap and setting up on-demand
 * CONTOUR conversion. Called from extension's onModeEnter.
 *
 * Flow:
 * 1. Extract StudyInstanceUID from URL
 * 2. Check if labelmap is available (call backend API)
 * 3. Fetch and parse NIfTI labelmap
 * 4. Create Cornerstone3D volume
 * 5. Add as LABELMAP segmentation
 * 6. Subscribe to visibility changes for on-demand CONTOUR conversion
 */

import {
  fetchLabelConfig,
  fetchNiftiLabelmap,
  getUniqueLabels,
} from './niftiLabelmapLoader';

import {
  initializeManager,
  createLabelmapVolume,
  addLabelmapSegmentation,
  setSegmentVisibility,
  subscribeToVisibilityChanges,
  cleanupManager,
  getManagerState,
} from './onDemandSegmentationManager';

// API base URL
const API_BASE_URL = '/api';

// Store unsubscribe functions
let visibilityUnsubscribe: (() => void) | null = null;

/**
 * Extract StudyInstanceUID from current URL
 */
function getStudyInstanceUIDFromURL(): string | null {
  const params = new URLSearchParams(window.location.search);
  return params.get('StudyInstanceUIDs');
}

/**
 * Check if labelmap is available for a study
 */
async function checkLabelmapAvailable(studyInstanceUID: string): Promise<boolean> {
  try {
    // Use GET instead of HEAD (FastAPI doesn't auto-support HEAD for GET endpoints)
    const response = await fetch(`${API_BASE_URL}/label-config/${studyInstanceUID}`);
    return response.ok;
  } catch {
    return false;
  }
}

/**
 * Wait for viewports to be ready
 */
async function waitForViewport(
  servicesManager: any,
  maxWaitMs: number = 10000
): Promise<string | null> {
  const { viewportGridService } = servicesManager.services;

  const startTime = Date.now();
  const checkInterval = 500;

  while (Date.now() - startTime < maxWaitMs) {
    // Try different methods to get viewport ID
    let viewportId: string | null = null;

    if (viewportGridService?.getActiveViewportId) {
      viewportId = viewportGridService.getActiveViewportId();
    }
    if (!viewportId && viewportGridService?.getState) {
      const state = viewportGridService.getState();
      viewportId = state?.activeViewportId;

      // If no active viewport, try first viewport
      if (!viewportId && state?.viewports?.size > 0) {
        viewportId = Array.from(state.viewports.keys())[0] as string;
      }
    }

    if (viewportId) {
      return viewportId;
    }

    await new Promise(resolve => setTimeout(resolve, checkInterval));
  }

  return null;
}

/**
 * Check if CT volume is currently loaded in viewport
 */
function isVolumeCurrentlyLoaded(
  servicesManager: any,
  viewportId: string
): { loaded: boolean; dimensions?: number[] } {
  try {
    const { cornerstoneViewportService } = servicesManager.services;
    const viewport = cornerstoneViewportService?.getCornerstoneViewport?.(viewportId);

    if (viewport) {
      // Check if viewport has volume actors with valid image data
      const actors = viewport.getActors?.();
      if (actors && actors.length > 0) {
        // For volume viewports, check if imageData exists
        const defaultActor = viewport.getDefaultActor?.();
        if (defaultActor?.actor) {
          const imageData = defaultActor.actor.getMapper?.()?.getInputData?.();
          if (imageData) {
            const dimensions = imageData.getDimensions?.();
            if (dimensions && dimensions[0] > 0) {
              return { loaded: true, dimensions };
            }
          }
        }

        // Alternative check: getImageData on viewport
        const viewportImageData = viewport.getImageData?.();
        if (viewportImageData?.dimensions && viewportImageData.dimensions[0] > 0) {
          return { loaded: true, dimensions: viewportImageData.dimensions };
        }
      }
    }
  } catch (e) {
    // Viewport not ready
  }
  return { loaded: false };
}

/**
 * Wait for CT volume to be fully loaded in the viewport
 * Uses hybrid approach: check current state first, then event listener
 */
async function waitForVolumeLoaded(
  servicesManager: any,
  viewportId: string,
  maxWaitMs: number = 30000
): Promise<boolean> {
  console.log(`[LabelmapIntegration] Waiting for CT volume to load in viewport: ${viewportId}`);

  // Step 1: Check if already loaded (handles case where event already fired)
  const currentState = isVolumeCurrentlyLoaded(servicesManager, viewportId);
  if (currentState.loaded) {
    console.log(`[LabelmapIntegration] CT volume already loaded: dimensions ${currentState.dimensions?.join('x')}`);
    return true;
  }

  // Step 2: Listen for IMAGE_VOLUME_LOADING_COMPLETED event
  try {
    // Use global Cornerstone instance (shared with OHIF) for event listening
    const cornerstone = (window as any).cornerstone || (window as any).cornerstoneCore;
    if (!cornerstone) {
      console.warn('[LabelmapIntegration] Cornerstone not found on window, falling back to polling');
      throw new Error('Cornerstone not available');
    }
    const { Enums, eventTarget } = cornerstone;

    return new Promise<boolean>((resolve) => {
      const timeout = setTimeout(() => {
        eventTarget.removeEventListener(Enums.Events.IMAGE_VOLUME_LOADING_COMPLETED, handler);
        console.warn('[LabelmapIntegration] Timeout waiting for CT volume to load');
        resolve(false);
      }, maxWaitMs);

      const handler = (evt: any) => {
        // Check if this is for our viewport's volume
        clearTimeout(timeout);
        eventTarget.removeEventListener(Enums.Events.IMAGE_VOLUME_LOADING_COMPLETED, handler);

        // Verify volume is actually ready
        const state = isVolumeCurrentlyLoaded(servicesManager, viewportId);
        if (state.loaded) {
          console.log(`[LabelmapIntegration] CT volume loaded (event): dimensions ${state.dimensions?.join('x')}`);
          resolve(true);
        } else {
          // Event fired but volume not ready for our viewport, keep waiting
          console.log('[LabelmapIntegration] Volume event received but not for our viewport, re-checking...');
          // Small delay then check again
          setTimeout(() => {
            const finalState = isVolumeCurrentlyLoaded(servicesManager, viewportId);
            resolve(finalState.loaded);
          }, 500);
        }
      };

      eventTarget.addEventListener(Enums.Events.IMAGE_VOLUME_LOADING_COMPLETED, handler);

      // Also check periodically in case event was missed or for different volume
      const checkInterval = setInterval(() => {
        const state = isVolumeCurrentlyLoaded(servicesManager, viewportId);
        if (state.loaded) {
          clearInterval(checkInterval);
          clearTimeout(timeout);
          eventTarget.removeEventListener(Enums.Events.IMAGE_VOLUME_LOADING_COMPLETED, handler);
          console.log(`[LabelmapIntegration] CT volume loaded (poll): dimensions ${state.dimensions?.join('x')}`);
          resolve(true);
        }
      }, 2000); // Backup polling every 2s, much less frequent

      // Clean up interval on timeout
      setTimeout(() => clearInterval(checkInterval), maxWaitMs);
    });
  } catch (e) {
    console.warn('[LabelmapIntegration] Event-based waiting failed, falling back:', e);
    // Fallback: simple polling
    const startTime = Date.now();
    while (Date.now() - startTime < maxWaitMs) {
      const state = isVolumeCurrentlyLoaded(servicesManager, viewportId);
      if (state.loaded) {
        console.log(`[LabelmapIntegration] CT volume loaded (fallback): dimensions ${state.dimensions?.join('x')}`);
        return true;
      }
      await new Promise(resolve => setTimeout(resolve, 1000));
    }
    return false;
  }
}

/**
 * Load labelmap for a study
 * This is the main function called from onModeEnter
 */
export async function loadLabelmapForStudy(
  servicesManager: any
): Promise<{
  success: boolean;
  segmentationId?: string;
  error?: string;
}> {
  // Check if we're in the right mode
  if (!window.location.pathname.includes('nerve-assessment')) {
    return { success: false, error: 'Not in nerve-assessment mode' };
  }

  // Get Study UID from URL
  const studyInstanceUID = getStudyInstanceUIDFromURL();
  if (!studyInstanceUID) {
    console.log('[LabelmapIntegration] No StudyInstanceUID in URL');
    return { success: false, error: 'No StudyInstanceUID' };
  }

  console.log(`[LabelmapIntegration] Loading labelmap for study: ${studyInstanceUID.slice(0, 20)}...`);

  try {
    // Step 1: Check if labelmap is available
    const isAvailable = await checkLabelmapAvailable(studyInstanceUID);
    if (!isAvailable) {
      console.log('[LabelmapIntegration] Labelmap not available (run nerve analysis first)');
      return { success: false, error: 'Labelmap not available' };
    }

    // Step 2: Fetch label configuration
    console.log('[LabelmapIntegration] Fetching label config...');
    const labelConfig = await fetchLabelConfig(studyInstanceUID, API_BASE_URL);

    // Step 3: Fetch and parse NIfTI labelmap
    console.log('[LabelmapIntegration] Fetching NIfTI labelmap...');
    const niftiData = await fetchNiftiLabelmap(studyInstanceUID, API_BASE_URL);

    // Log unique labels found
    const uniqueLabels = getUniqueLabels(niftiData.typedArray);
    console.log(`[LabelmapIntegration] Found ${uniqueLabels.length} unique labels:`, uniqueLabels);

    // Step 4: Wait for viewport to be ready
    console.log('[LabelmapIntegration] Waiting for viewport...');
    const viewportId = await waitForViewport(servicesManager);
    if (!viewportId) {
      console.warn('[LabelmapIntegration] Viewport not available');
      return { success: false, error: 'Viewport not ready' };
    }
    console.log(`[LabelmapIntegration] Viewport ready: ${viewportId}`);

    // Step 4.5: Wait for CT volume to be fully loaded
    // This is critical - createLocalVolume needs reference volume data
    console.log('[LabelmapIntegration] Waiting for CT volume to load...');
    const volumeLoaded = await waitForVolumeLoaded(servicesManager, viewportId);
    if (!volumeLoaded) {
      console.warn('[LabelmapIntegration] CT volume not loaded, proceeding anyway...');
      // Don't return error - try to create labelmap anyway, it might work
    }

    // Step 5: Create labelmap volume
    console.log('[LabelmapIntegration] Creating labelmap volume...');
    const volumeId = await createLabelmapVolume(
      niftiData,
      labelConfig,
      studyInstanceUID,
      servicesManager
    );

    // Step 6: Add as segmentation
    console.log('[LabelmapIntegration] Adding segmentation...');
    const segmentationId = await addLabelmapSegmentation(
      volumeId,
      labelConfig,
      studyInstanceUID,
      servicesManager
    );

    // Step 7: Subscribe to visibility changes for on-demand conversion
    console.log('[LabelmapIntegration] Setting up on-demand conversion...');
    visibilityUnsubscribe = subscribeToVisibilityChanges(
      servicesManager,
      studyInstanceUID,
      viewportId
    );

    // Step 8: Auto-show first few important segments (nerves)
    await autoShowImportantSegments(
      labelConfig,
      studyInstanceUID,
      viewportId,
      servicesManager
    );

    console.log('[LabelmapIntegration] Labelmap loaded successfully!');

    return {
      success: true,
      segmentationId,
    };
  } catch (error: any) {
    console.error('[LabelmapIntegration] Failed to load labelmap:', error);
    return {
      success: false,
      error: error?.message || 'Unknown error',
    };
  }
}

/**
 * Auto-show important segments (nerves) on load
 * Only show a few to avoid memory issues
 */
async function autoShowImportantSegments(
  labelConfig: any,
  studyInstanceUID: string,
  viewportId: string,
  servicesManager: any
): Promise<void> {
  // Priority patterns for auto-show (nerves first)
  const priorityPatterns = [
    /nerve/i,
    /vagus/i,
    /phrenic/i,
    /rln/i,
    /ebsln/i,
  ];

  const maxAutoShow = 4; // Max segments to auto-show
  let shown = 0;

  for (const segment of labelConfig.segments) {
    if (shown >= maxAutoShow) break;

    // Check if segment matches priority patterns
    const isPriority = priorityPatterns.some(pattern => pattern.test(segment.label));

    if (isPriority) {
      console.log(`[LabelmapIntegration] Auto-showing segment: ${segment.label}`);
      await setSegmentVisibility(
        segment.label,
        true,
        studyInstanceUID,
        viewportId,
        servicesManager
      );
      shown++;
    }
  }

  if (shown === 0 && labelConfig.segments.length > 0) {
    // If no priority segments, show first one
    const firstSegment = labelConfig.segments[0];
    console.log(`[LabelmapIntegration] No priority segments, showing first: ${firstSegment.label}`);
    await setSegmentVisibility(
      firstSegment.label,
      true,
      studyInstanceUID,
      viewportId,
      servicesManager
    );
  }
}

/**
 * Cleanup labelmap resources
 * Called from onModeExit
 */
export function cleanupLabelmap(): void {
  console.log('[LabelmapIntegration] Cleaning up...');

  // Unsubscribe from visibility changes
  if (visibilityUnsubscribe) {
    visibilityUnsubscribe();
    visibilityUnsubscribe = null;
  }

  // Get study UID and cleanup manager
  const studyInstanceUID = getStudyInstanceUIDFromURL();
  if (studyInstanceUID) {
    cleanupManager(studyInstanceUID);
  }
}

/**
 * Check if labelmap is loaded for current study
 */
export function isLabelmapLoaded(): boolean {
  const studyInstanceUID = getStudyInstanceUIDFromURL();
  if (!studyInstanceUID) return false;

  const state = getManagerState(studyInstanceUID);
  return state?.isLoaded ?? false;
}

/**
 * Get current labelmap segment states
 */
export function getLabelmapSegments(): any[] {
  const studyInstanceUID = getStudyInstanceUIDFromURL();
  if (!studyInstanceUID) return [];

  const state = getManagerState(studyInstanceUID);
  if (!state) return [];

  return Array.from(state.segments.values());
}

export default {
  loadLabelmapForStudy,
  cleanupLabelmap,
  isLabelmapLoaded,
  getLabelmapSegments,
};

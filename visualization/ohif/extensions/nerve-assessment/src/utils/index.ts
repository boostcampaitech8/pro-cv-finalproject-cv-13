/**
 * Utility functions for NIfTI Labelmap handling
 *
 * On-Demand CONTOUR Conversion Strategy:
 * - NIfTI labelmap provides better sagittal/coronal support than RTSTRUCT
 * - Only visible segments are converted to CONTOUR (not all 22+ at once)
 * - Avoids browser memory crash (~1.5GB+ if all converted simultaneously)
 */

export * from './niftiLabelmapLoader';
export * from './onDemandSegmentationManager';
export * from './labelmapIntegration';
export * from './surfaceLoader';

// Default exports
export { default as niftiLabelmapLoader } from './niftiLabelmapLoader';
export { default as onDemandSegmentationManager } from './onDemandSegmentationManager';
export { default as labelmapIntegration } from './labelmapIntegration';

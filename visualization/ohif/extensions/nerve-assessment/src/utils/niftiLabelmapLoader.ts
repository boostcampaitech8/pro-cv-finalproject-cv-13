/**
 * NIfTI Labelmap Loader for Cornerstone3D
 *
 * Loads multi-label NIfTI from backend API and creates a LABELMAP segmentation.
 * Uses on-demand CONTOUR conversion - only visible segments are converted to avoid
 * browser memory issues with 22+ segments.
 *
 * Flow:
 * 1. Fetch .nii.gz from /api/nifti-labelmap/{studyUID}
 * 2. Decompress with pako
 * 3. Parse NIfTI header (dimensions, datatype, affine)
 * 4. Create Cornerstone3D volume from raw data
 * 5. Add as LABELMAP segmentation
 * 6. On-demand: Convert visible segments to CONTOUR
 */

import pako from 'pako';

// NIfTI header constants
const NIFTI1_HEADER_SIZE = 348;
const NIFTI1_MAGIC = 'n+1\0';
const NIFTI1_MAGIC_ALT = 'ni1\0';

// NIfTI datatype codes
const NIFTI_TYPE_UINT8 = 2;
const NIFTI_TYPE_INT16 = 4;
const NIFTI_TYPE_INT32 = 8;
const NIFTI_TYPE_FLOAT32 = 16;
const NIFTI_TYPE_FLOAT64 = 64;
const NIFTI_TYPE_UINT16 = 512;
const NIFTI_TYPE_UINT32 = 768;

/**
 * Label configuration from backend
 */
export interface LabelConfig {
  labels: Record<string, number>; // structure_name -> label_id
  colors: Record<string, number[]>; // label_id -> [R, G, B, A]
  segments: Array<{
    segmentIndex: number;
    label: string;
    color: number[];
  }>;
  metadata?: {
    study_uid: string;
    reference_nifti: string;
  };
}

/**
 * NIfTI header information
 */
interface NiftiHeader {
  dims: number[]; // [ndim, dim1, dim2, dim3, dim4, ...]
  datatype: number;
  bitpix: number;
  pixdim: number[]; // voxel dimensions (spacing)
  vox_offset: number; // offset to data
  scl_slope: number;
  scl_inter: number;
  qform_code: number;
  sform_code: number;
  quatern_b: number;
  quatern_c: number;
  quatern_d: number;
  qoffset_x: number;
  qoffset_y: number;
  qoffset_z: number;
  srow_x: number[];
  srow_y: number[];
  srow_z: number[];
}

/**
 * Parsed NIfTI data
 */
export interface NiftiData {
  header: NiftiHeader;
  data: ArrayBuffer;
  typedArray: Uint8Array | Int16Array | Int32Array | Float32Array | Float64Array | Uint16Array | Uint32Array;
  dimensions: [number, number, number];
  spacing: [number, number, number];
  origin: [number, number, number];
  direction: number[];
}

/**
 * Decompress gzipped data using pako
 */
function decompressGzip(data: ArrayBuffer): ArrayBuffer {
  try {
    const compressed = new Uint8Array(data);
    const decompressed = pako.inflate(compressed);
    return decompressed.buffer;
  } catch (error) {
    console.error('[NiftiLoader] Decompression failed:', error);
    throw new Error('Failed to decompress NIfTI file');
  }
}

/**
 * Parse NIfTI-1 header from ArrayBuffer
 */
function parseNiftiHeader(buffer: ArrayBuffer): NiftiHeader {
  const view = new DataView(buffer);

  // Check magic number (bytes 344-347)
  const magic = String.fromCharCode(
    view.getUint8(344),
    view.getUint8(345),
    view.getUint8(346),
    view.getUint8(347)
  );

  if (magic !== NIFTI1_MAGIC && magic !== NIFTI1_MAGIC_ALT) {
    throw new Error(`Invalid NIfTI magic: ${magic}`);
  }

  // Determine endianness from dim[0] (should be 1-7)
  let littleEndian = true;
  const dim0 = view.getInt16(40, true);
  if (dim0 < 1 || dim0 > 7) {
    littleEndian = false;
  }

  // Parse dimensions (bytes 40-55)
  const dims: number[] = [];
  for (let i = 0; i < 8; i++) {
    dims.push(view.getInt16(40 + i * 2, littleEndian));
  }

  // Parse datatype and bitpix
  const datatype = view.getInt16(70, littleEndian);
  const bitpix = view.getInt16(72, littleEndian);

  // Parse pixel dimensions (spacing)
  const pixdim: number[] = [];
  for (let i = 0; i < 8; i++) {
    pixdim.push(view.getFloat32(76 + i * 4, littleEndian));
  }

  // Parse voxel offset
  const vox_offset = view.getFloat32(108, littleEndian);

  // Parse scaling
  const scl_slope = view.getFloat32(112, littleEndian);
  const scl_inter = view.getFloat32(116, littleEndian);

  // Parse qform/sform codes
  const qform_code = view.getInt16(252, littleEndian);
  const sform_code = view.getInt16(254, littleEndian);

  // Parse quaternion parameters
  const quatern_b = view.getFloat32(256, littleEndian);
  const quatern_c = view.getFloat32(260, littleEndian);
  const quatern_d = view.getFloat32(264, littleEndian);
  const qoffset_x = view.getFloat32(268, littleEndian);
  const qoffset_y = view.getFloat32(272, littleEndian);
  const qoffset_z = view.getFloat32(276, littleEndian);

  // Parse sform matrix rows
  const srow_x: number[] = [];
  const srow_y: number[] = [];
  const srow_z: number[] = [];
  for (let i = 0; i < 4; i++) {
    srow_x.push(view.getFloat32(280 + i * 4, littleEndian));
    srow_y.push(view.getFloat32(296 + i * 4, littleEndian));
    srow_z.push(view.getFloat32(312 + i * 4, littleEndian));
  }

  return {
    dims,
    datatype,
    bitpix,
    pixdim,
    vox_offset,
    scl_slope,
    scl_inter,
    qform_code,
    sform_code,
    quatern_b,
    quatern_c,
    quatern_d,
    qoffset_x,
    qoffset_y,
    qoffset_z,
    srow_x,
    srow_y,
    srow_z,
  };
}

/**
 * Create typed array from NIfTI data based on datatype
 */
function createTypedArray(
  buffer: ArrayBuffer,
  offset: number,
  datatype: number,
  numVoxels: number
): NiftiData['typedArray'] {
  const dataBuffer = buffer.slice(offset);

  switch (datatype) {
    case NIFTI_TYPE_UINT8:
      return new Uint8Array(dataBuffer, 0, numVoxels);
    case NIFTI_TYPE_INT16:
      return new Int16Array(dataBuffer, 0, numVoxels);
    case NIFTI_TYPE_INT32:
      return new Int32Array(dataBuffer, 0, numVoxels);
    case NIFTI_TYPE_FLOAT32:
      return new Float32Array(dataBuffer, 0, numVoxels);
    case NIFTI_TYPE_FLOAT64:
      return new Float64Array(dataBuffer, 0, numVoxels);
    case NIFTI_TYPE_UINT16:
      return new Uint16Array(dataBuffer, 0, numVoxels);
    case NIFTI_TYPE_UINT32:
      return new Uint32Array(dataBuffer, 0, numVoxels);
    default:
      console.warn(`[NiftiLoader] Unknown datatype ${datatype}, using Uint8`);
      return new Uint8Array(dataBuffer, 0, numVoxels);
  }
}

/**
 * Calculate origin from sform or qform matrix.
 * Applies RAS→LPS conversion: NIfTI uses RAS, Cornerstone3D uses LPS.
 * Negate X (Right→Left) and Y (Anterior→Posterior), keep Z (Superior).
 */
function calculateOrigin(header: NiftiHeader): [number, number, number] {
  if (header.sform_code > 0) {
    return [-header.srow_x[3], -header.srow_y[3], header.srow_z[3]];
  }
  if (header.qform_code > 0) {
    return [-header.qoffset_x, -header.qoffset_y, header.qoffset_z];
  }
  return [0, 0, 0];
}

/**
 * Calculate direction cosines from sform matrix.
 * Applies RAS→LPS conversion: negate X and Y rows of the rotation matrix.
 */
function calculateDirection(header: NiftiHeader): number[] {
  if (header.sform_code > 0) {
    const sx = header.srow_x; // X (Right in RAS) components → negate for LPS
    const sy = header.srow_y; // Y (Anterior in RAS) components → negate for LPS
    const sz = header.srow_z; // Z (Superior) components → keep

    // Normalize direction vectors
    const spacing = [header.pixdim[1], header.pixdim[2], header.pixdim[3]];

    return [
      -sx[0] / spacing[0], -sy[0] / spacing[0], sz[0] / spacing[0],
      -sx[1] / spacing[1], -sy[1] / spacing[1], sz[1] / spacing[1],
      -sx[2] / spacing[2], -sy[2] / spacing[2], sz[2] / spacing[2],
    ];
  }

  // Identity direction if no transform
  return [1, 0, 0, 0, 1, 0, 0, 0, 1];
}

/**
 * Parse NIfTI file from ArrayBuffer
 */
export function parseNifti(buffer: ArrayBuffer): NiftiData {
  // Check if gzipped (magic bytes 1f 8b)
  const firstBytes = new Uint8Array(buffer, 0, 2);
  let decompressedBuffer = buffer;

  if (firstBytes[0] === 0x1f && firstBytes[1] === 0x8b) {
    console.log('[NiftiLoader] Decompressing gzipped NIfTI...');
    decompressedBuffer = decompressGzip(buffer);
  }

  // Parse header
  const header = parseNiftiHeader(decompressedBuffer);

  console.log('[NiftiLoader] NIfTI header parsed:', {
    dims: header.dims.slice(0, 4),
    datatype: header.datatype,
    spacing: header.pixdim.slice(1, 4),
    vox_offset: header.vox_offset,
  });

  // Calculate dimensions
  const dimensions: [number, number, number] = [
    header.dims[1],
    header.dims[2],
    header.dims[3],
  ];
  const numVoxels = dimensions[0] * dimensions[1] * dimensions[2];

  // Create typed array
  const offset = Math.max(header.vox_offset, NIFTI1_HEADER_SIZE);
  const typedArray = createTypedArray(decompressedBuffer, offset, header.datatype, numVoxels);

  // Calculate geometry
  const spacing: [number, number, number] = [
    Math.abs(header.pixdim[1]) || 1,
    Math.abs(header.pixdim[2]) || 1,
    Math.abs(header.pixdim[3]) || 1,
  ];
  const origin = calculateOrigin(header);
  const direction = calculateDirection(header);

  return {
    header,
    data: decompressedBuffer,
    typedArray,
    dimensions,
    spacing,
    origin,
    direction,
  };
}

/**
 * Fetch label configuration from backend
 */
export async function fetchLabelConfig(
  studyInstanceUID: string,
  apiBaseUrl: string = '/api'
): Promise<LabelConfig> {
  const url = `${apiBaseUrl}/label-config/${studyInstanceUID}`;

  console.log(`[NiftiLoader] Fetching label config: ${url}`);

  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`Failed to fetch label config: ${response.status} ${response.statusText}`);
  }

  const config = await response.json();
  console.log('[NiftiLoader] Label config loaded:', {
    numLabels: Object.keys(config.labels || {}).length,
    segments: config.segments?.length || 0,
  });

  return config;
}

/**
 * Fetch and parse NIfTI labelmap from backend
 */
export async function fetchNiftiLabelmap(
  studyInstanceUID: string,
  apiBaseUrl: string = '/api'
): Promise<NiftiData> {
  const url = `${apiBaseUrl}/nifti-labelmap/${studyInstanceUID}`;

  console.log(`[NiftiLoader] Fetching NIfTI labelmap: ${url}`);

  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`Failed to fetch NIfTI labelmap: ${response.status} ${response.statusText}`);
  }

  const arrayBuffer = await response.arrayBuffer();
  console.log(`[NiftiLoader] Downloaded ${(arrayBuffer.byteLength / 1024 / 1024).toFixed(2)} MB`);

  return parseNifti(arrayBuffer);
}

/**
 * Get unique labels from labelmap data
 */
export function getUniqueLabels(typedArray: NiftiData['typedArray']): number[] {
  const labels = new Set<number>();

  for (let i = 0; i < typedArray.length; i++) {
    const value = typedArray[i];
    if (value > 0) {
      labels.add(value);
    }
  }

  return Array.from(labels).sort((a, b) => a - b);
}

/**
 * Create a binary mask for a specific label
 */
export function extractLabelMask(
  typedArray: NiftiData['typedArray'],
  labelIndex: number
): Uint8Array {
  const mask = new Uint8Array(typedArray.length);

  for (let i = 0; i < typedArray.length; i++) {
    mask[i] = typedArray[i] === labelIndex ? 1 : 0;
  }

  return mask;
}

export default {
  parseNifti,
  fetchLabelConfig,
  fetchNiftiLabelmap,
  getUniqueLabels,
  extractLabelMask,
};

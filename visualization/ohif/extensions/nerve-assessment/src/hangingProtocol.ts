/**
 * Custom Hanging Protocol for Nerve Assessment Mode - MPR + 3D Layout
 *
 * Features:
 * - 2x2 grid layout (Axial, Sagittal, Coronal, 3D)
 * - Volume viewports for MPR reconstruction
 * - Volume3D viewport for Surface mesh rendering
 * - Crosshairs tool support (requires 2+ viewports)
 * - RTSS contour overlay via hydrateseg syncGroup
 * - VOI synchronization across viewports
 */

// Sync group for automatic RTSS/SEG hydration across viewports
const HYDRATE_SEG_SYNC_GROUP = {
  type: 'hydrateseg',
  id: 'sameFORId',
  source: true,
  target: true,
  options: {
    matchingRules: ['sameFOR'],
  },
};

// Sync group for VOI (Window/Level) synchronization
const VOI_SYNC_GROUP = {
  type: 'voi',
  id: 'mpr',
  source: true,
  target: true,
};

const nerveAssessmentHangingProtocol = {
  id: 'nerve-assessment-mpr',
  description: 'Nerve Assessment MPR + 3D Mode - 4 Viewport Layout',
  name: 'Nerve Assessment MPR + 3D',

  protocolMatchingRules: [
    {
      id: 'OneOrMoreSeries',
      weight: 25,
      attribute: 'numberOfDisplaySetsWithImages',
      constraint: {
        greaterThan: 0,
      },
    },
  ],

  // Use 'mpr' tool group for MPR-specific tools including Crosshairs
  toolGroupIds: ['mpr'],

  displaySetSelectors: {
    activeDisplaySet: {
      allowUnmatchedView: true,
      seriesMatchingRules: [
        {
          attribute: 'numImageFrames',
          constraint: {
            greaterThan: { value: 0 },
          },
          weight: 1,
          required: true,
        },
        {
          attribute: 'isDisplaySetFromUrl',
          weight: 20,
          constraint: {
            equals: true,
          },
        },
        // Prefer CT modality
        {
          attribute: 'Modality',
          constraint: {
            equals: 'CT',
          },
          weight: 10,
        },
      ],
    },
  },

  defaultViewport: {
    viewportOptions: {
      viewportType: 'volume',
      toolGroupId: 'mpr',
      syncGroups: [VOI_SYNC_GROUP, HYDRATE_SEG_SYNC_GROUP],
    },
    displaySets: [
      {
        id: 'activeDisplaySet',
        matchedDisplaySetsIndex: -1,
      },
    ],
  },

  stages: [
    {
      id: 'mpr-3d-2x2',
      name: 'MPR + 3D 2x2',
      stageActivation: {
        enabled: {
          minViewportsMatched: 1,
        },
      },
      viewportStructure: {
        layoutType: 'grid',
        properties: {
          rows: 2,
          columns: 2,  // 4 viewports: Axial, Sagittal, Coronal, 3D
        },
      },
      viewports: [
        // Axial viewport (top-left)
        {
          viewportOptions: {
            viewportId: 'mpr-axial',
            viewportType: 'volume',
            orientation: 'axial',
            toolGroupId: 'mpr',
            allowUnmatchedView: true,
            syncGroups: [VOI_SYNC_GROUP, HYDRATE_SEG_SYNC_GROUP],
          },
          displaySets: [
            {
              id: 'activeDisplaySet',
            },
          ],
        },
        // Sagittal viewport (top-right)
        {
          viewportOptions: {
            viewportId: 'mpr-sagittal',
            viewportType: 'volume',
            orientation: 'sagittal',
            toolGroupId: 'mpr',
            allowUnmatchedView: true,
            syncGroups: [VOI_SYNC_GROUP, HYDRATE_SEG_SYNC_GROUP],
          },
          displaySets: [
            {
              id: 'activeDisplaySet',
            },
          ],
        },
        // Coronal viewport (bottom-left)
        {
          viewportOptions: {
            viewportId: 'mpr-coronal',
            viewportType: 'volume',
            orientation: 'coronal',
            toolGroupId: 'mpr',
            allowUnmatchedView: true,
            syncGroups: [VOI_SYNC_GROUP, HYDRATE_SEG_SYNC_GROUP],
          },
          displaySets: [
            {
              id: 'activeDisplaySet',
            },
          ],
        },
        // 3D viewport (bottom-right) - for Surface mesh rendering
        {
          viewportOptions: {
            viewportId: 'volume3d',
            viewportType: 'volume3d',
            toolGroupId: 'volume3d',
            allowUnmatchedView: true,
            // No syncGroups for 3D - independent rendering
          },
          displaySets: [
            {
              id: 'activeDisplaySet',
            },
          ],
        },
      ],
    },
  ],

  numberOfPriorsReferenced: -1,
};

export default nerveAssessmentHangingProtocol;

window.config = {
  routerBasename: '/',
  showStudyList: true,  // Enable default study list with DICOM upload
  // defaultMode removed - modes are entered when viewing a study, not on study list
  extensions: [],
  modes: [],
  maxNumberOfWebWorkers: 3,
  omitQuotationForMultipartRequest: true,
  showWarningMessageForCrossOrigin: false,
  showCPUFallbackMessage: true,
  showLoadingIndicator: true,
  strictZSpacingForVolumeViewport: true,
  useSharedArrayBuffer: 'AUTO',
  maxNumRequests: {
    interaction: 100,
    thumbnail: 75,
    prefetch: 25,
  },
  // CustomizationService - empty object is the default for v3.11.0
  customizationService: {},
  whiteLabeling: {
    createLogoComponentFn: function(React) {
      return React.createElement('span', {
        style: {
          color: '#10b981',
          fontSize: '16px',
          fontWeight: 'bold',
          padding: '0 12px',
        },
      }, 'Team NextCT');
    },
  },
  dataSources: [
    {
      namespace: '@ohif/extension-default.dataSourcesModule.dicomweb',
      sourceName: 'dicomweb',
      configuration: {
        friendlyName: 'Orthanc DICOM Server',
        name: 'orthanc',
        wadoUriRoot: '/wado',
        qidoRoot: '/dicom-web',
        wadoRoot: '/dicom-web',
        qidoSupportsIncludeField: false,
        supportsReject: false,
        imageRendering: 'wadors',
        thumbnailRendering: 'wadors',
        enableStudyLazyLoad: true,
        supportsFuzzyMatching: false,
        supportsWildcard: true,
        staticWado: true,
        singlepart: 'bulkdata,video,pdf',
        bulkDataURI: {
          enabled: false,
        },
        omitQuotationForMultipartRequest: true,
        retrieveOptions: {
          default: {
            framesMode: 'singlepart',
          },
          multipleFast: {
            framesMode: 'singlepart',
          },
          single: {
            framesMode: 'singlepart',
          },
        },
        // Enable DICOM upload via STOW-RS
        dicomUploadEnabled: true,
        supportsStow: true,
        stowRoot: '/dicom-web',
      },
    },
  ],
  defaultDataSourceName: 'dicomweb',
  hotkeys: [
    { commandName: 'incrementActiveViewport', label: 'Next Viewport', keys: ['right'] },
    { commandName: 'decrementActiveViewport', label: 'Previous Viewport', keys: ['left'] },
    { commandName: 'rotateViewportCW', label: 'Rotate Right', keys: ['r'] },
    { commandName: 'rotateViewportCCW', label: 'Rotate Left', keys: ['l'] },
    { commandName: 'invertViewport', label: 'Invert', keys: ['i'] },
    { commandName: 'flipViewportHorizontal', label: 'Flip Horizontally', keys: ['h'] },
    { commandName: 'flipViewportVertical', label: 'Flip Vertically', keys: ['v'] },
    { commandName: 'scaleUpViewport', label: 'Zoom In', keys: ['+'] },
    { commandName: 'scaleDownViewport', label: 'Zoom Out', keys: ['-'] },
    { commandName: 'fitViewportToWindow', label: 'Zoom to Fit', keys: ['='] },
    { commandName: 'resetViewport', label: 'Reset', keys: ['space'] },
    { commandName: 'previousImage', label: 'Previous Image', keys: ['up'] },
    { commandName: 'nextImage', label: 'Next Image', keys: ['down'] },
  ],
  investigationalUseDialog: {
    option: 'never',
  },
};

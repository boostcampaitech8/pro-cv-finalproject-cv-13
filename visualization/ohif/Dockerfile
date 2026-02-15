# OHIF Viewer Dockerfile with Nerve Assessment Extension & Mode
FROM node:18-alpine AS builder

WORKDIR /app

# Install git
RUN apk add --no-cache git

# Clone OHIF Viewer v3.11.0 (stable with CONTOUR representation support via polyseg)
# v3.11.0 includes: Cornerstone3D v3.32.5 + @cornerstonejs/polymorphic-segmentation
RUN git clone --depth 1 --branch v3.11.0 https://github.com/OHIF/Viewers.git .

# Copy our custom extension
COPY extensions/nerve-assessment /app/extensions/nerve-assessment

# Copy our custom mode
COPY modes/nerve-assessment-mode /app/modes/nerve-assessment-mode

# Update root package.json to include our extension and mode in workspaces
RUN node -e " \
const fs = require('fs'); \
const pkgPath = 'package.json'; \
const pkg = JSON.parse(fs.readFileSync(pkgPath, 'utf8')); \
if (!pkg.workspaces.packages.includes('extensions/nerve-assessment')) { \
  pkg.workspaces.packages.push('extensions/nerve-assessment'); \
} \
if (!pkg.workspaces.packages.includes('modes/nerve-assessment-mode')) { \
  pkg.workspaces.packages.push('modes/nerve-assessment-mode'); \
} \
fs.writeFileSync(pkgPath, JSON.stringify(pkg, null, 2)); \
console.log('Updated package.json workspaces'); \
"

# Copy custom OHIF configuration
COPY config/default.js /app/platform/app/public/config/default.js

# Install all dependencies first
RUN yarn install

# CRITICAL: Patch polymorphic-segmentation to fix sagittal contour rendering
# The isPolylineParallelToPlane check incorrectly rejects thin/linear contours
# when slicing surfaces for sagittal views (Y-range is narrow)
RUN sed -i 's/if (!isPolylineParallelToPlane(lines, points, planeNormal))/if (false \&\& !isPolylineParallelToPlane(lines, points, planeNormal))/' \
    node_modules/@cornerstonejs/polymorphic-segmentation/dist/esm/Contour/utils/createAndAddContourSegmentationsFromClippedSurfaces.js && \
    echo "Patched isPolylineParallelToPlane check for sagittal rendering"

# Verify the patch was applied
RUN grep -n "if (false && !isPolylineParallelToPlane" node_modules/@cornerstonejs/polymorphic-segmentation/dist/esm/Contour/utils/createAndAddContourSegmentationsFromClippedSurfaces.js && \
    echo "Patch verified successfully"

# CRITICAL: Add Surface representation disable to cornerstone init.tsx
# This prevents memory errors when polySeg tries to convert contours to surfaces in 3D viewports
RUN node -e " \
const fs = require('fs'); \
const initPath = '/app/extensions/cornerstone/src/init.tsx'; \
let content = fs.readFileSync(initPath, 'utf8'); \
\
/* Check if already patched */ \
if (content.includes('SURFACE')) { \
  console.log('Surface disable already in init.tsx'); \
  process.exit(0); \
} \
\
/* Find the Contour style block and add Surface disable after it */ \
const contourStylePattern = /cornerstoneTools\\.segmentation\\.config\\.style\\.setStyle\\([\\s\\S]*?renderFill: true,[\\s\\S]*?fillAlpha: 0\\.3,[\\s\\S]*?\\}[\\s\\S]*?\\);/; \
const match = content.match(contourStylePattern); \
if (match) { \
  const surfaceDisable = '\\n\\n  // Disable automatic Surface representation for 3D viewports\\n  // Prevents memory errors when polySeg tries to convert contours to surfaces\\n  if (cornerstoneTools.segmentation.config.setGlobalConfig) {\\n    cornerstoneTools.segmentation.config.setGlobalConfig({\\n      representations: {\\n        SURFACE: {\\n          enabled: false,\\n        },\\n      },\\n    });\\n  }'; \
  content = content.replace(match[0], match[0] + surfaceDisable); \
  fs.writeFileSync(initPath, content); \
  console.log('SUCCESS: Added Surface disable to init.tsx'); \
} else { \
  console.log('WARNING: Could not find Contour style block in init.tsx'); \
} \
"

# Verify Surface disable patch
RUN grep -A5 "SURFACE" /app/extensions/cornerstone/src/init.tsx || echo "Surface config not found"

# CRITICAL: Expose polySeg on window for nerve-assessment extension
# This avoids module instance mismatch when extension uses dynamic import
RUN node -e " \
const fs = require('fs'); \
const filePath = '/app/extensions/cornerstone/src/initCornerstoneTools.js'; \
let content = fs.readFileSync(filePath, 'utf8'); \
\
/* Check if already patched */ \
if (content.includes('window.cornerstonePolySeg')) { \
  console.log('window.cornerstonePolySeg already exposed'); \
  process.exit(0); \
} \
\
/* Find polySeg import and add window exposure after it */ \
const importPattern = /import \\* as polySeg from '@cornerstonejs\\/polymorphic-segmentation';/; \
if (importPattern.test(content)) { \
  content = content.replace( \
    importPattern, \
    \"import * as polySeg from '@cornerstonejs/polymorphic-segmentation';\\n\\n// Expose polySeg on window for extensions to access (avoids module instance mismatch)\\nwindow.cornerstonePolySeg = polySeg;\" \
  ); \
  fs.writeFileSync(filePath, content); \
  console.log('SUCCESS: Exposed polySeg on window'); \
} else { \
  console.log('WARNING: Could not find polySeg import'); \
} \
"

# Verify polySeg window exposure
RUN grep -n "window.cornerstonePolySeg" /app/extensions/cornerstone/src/initCornerstoneTools.js || echo "polySeg window exposure not found"

# CRITICAL: Add polySeg.init() call to register web workers
# cornerstoneTools.init() does NOT auto-call addon's init - only setConfig
# polySeg.init() calls registerPolySegWorker() to register 'polySeg' worker
RUN node -e " \
const fs = require('fs'); \
const filePath = '/app/extensions/cornerstone/src/initCornerstoneTools.js'; \
let content = fs.readFileSync(filePath, 'utf8'); \
\
/* Check if already patched */ \
if (content.includes('polySeg.init()')) { \
  console.log('polySeg.init() already added'); \
  process.exit(0); \
} \
\
/* Find init({ addons: { polySeg } }) and add polySeg.init() after it */ \
const initPattern = /init\\(\\{\\s*addons:\\s*\\{\\s*polySeg\\s*\\}\\s*\\}\\);?/; \
if (initPattern.test(content)) { \
  content = content.replace( \
    initPattern, \
    'init({ addons: { polySeg } });\\n\\n  // Register polySeg web workers (must be called explicitly)\\n  polySeg.init();' \
  ); \
  fs.writeFileSync(filePath, content); \
  console.log('SUCCESS: Added polySeg.init() call'); \
} else { \
  /* Fallback: try to find init call with different formatting */ \
  const altPattern = /init\\(\\{[\\s\\S]*?addons[\\s\\S]*?polySeg[\\s\\S]*?\\}\\);?/; \
  const match = content.match(altPattern); \
  if (match) { \
    content = content.replace(match[0], match[0] + '\\n\\n  // Register polySeg web workers\\n  polySeg.init();'); \
    fs.writeFileSync(filePath, content); \
    console.log('SUCCESS: Added polySeg.init() (alt pattern)'); \
  } else { \
    console.log('WARNING: Could not find init({ addons: { polySeg } }) pattern'); \
  } \
} \
"

# Verify polySeg.init() patch
RUN grep -n "polySeg.init()" /app/extensions/cornerstone/src/initCornerstoneTools.js || echo "polySeg.init() not found"

# Use yarn link for the packages (needed for webpack resolution)
RUN yarn run cli link-extension /app/extensions/nerve-assessment 2>&1 || true
RUN yarn run cli link-mode /app/modes/nerve-assessment-mode 2>&1 || true

# Modify pluginConfig.json - enable required extensions and add our extension/mode
RUN node -e " \
const fs = require('fs'); \
const configPath = '/app/platform/app/pluginConfig.json'; \
const config = JSON.parse(fs.readFileSync(configPath, 'utf8')); \
\
/* Enable dicom-seg extension (remove default: false so it gets loaded) */ \
const segExt = config.extensions.find(e => e.packageName === '@ohif/extension-cornerstone-dicom-seg'); \
if (segExt && segExt.default === false) { \
  delete segExt.default; \
  console.log('Enabled @ohif/extension-cornerstone-dicom-seg (removed default: false)'); \
} \
\
/* Enable dicom-rt extension */ \
const rtExt = config.extensions.find(e => e.packageName === '@ohif/extension-cornerstone-dicom-rt'); \
if (rtExt && rtExt.default === false) { \
  delete rtExt.default; \
  console.log('Enabled @ohif/extension-cornerstone-dicom-rt (removed default: false)'); \
} \
\
/* Add our extension if not present */ \
const extExists = config.extensions.some(e => e.packageName === '@ohif/extension-nerve-assessment'); \
if (!extExists) { \
  config.extensions.push({ packageName: '@ohif/extension-nerve-assessment' }); \
  console.log('Added @ohif/extension-nerve-assessment to extensions'); \
} \
\
/* Add our mode if not present */ \
const modeExists = config.modes.some(m => m.packageName === '@ohif/mode-nerve-assessment'); \
if (!modeExists) { \
  config.modes.push({ packageName: '@ohif/mode-nerve-assessment' }); \
  console.log('Added @ohif/mode-nerve-assessment to modes'); \
} \
\
/* Remove unused default modes */ \
const modesToRemove = ['@ohif/mode-tmtv', '@ohif/mode-microscopy', '@ohif/mode-preclinical-4d', '@ohif/mode-ultrasound-pleura-bline']; \
config.modes = config.modes.filter(m => !modesToRemove.includes(m.packageName)); \
console.log('Removed unused modes:', modesToRemove.join(', ')); \
\
fs.writeFileSync(configPath, JSON.stringify(config, null, 2)); \
console.log('Updated pluginConfig.json'); \
"

# Debug: Verify extensions and mode are in the config
RUN echo "=== Final pluginConfig.json ===" && \
    cat /app/platform/app/pluginConfig.json

# CRITICAL: Delete ALL caches (webpack, nx, lerna)
RUN rm -rf /app/node_modules/.cache && \
    rm -rf /app/.nx && \
    rm -rf /app/node_modules/.cache/nx && \
    rm -rf /root/.cache/nx && \
    rm -rf /root/.nx && \
    echo "Deleted all caches"

# Manually run writePluginImportsFile to generate pluginImports.js
RUN cd /app/platform/app && node -e " \
const writePluginImportsFile = require('./.webpack/writePluginImportsFile.js'); \
writePluginImportsFile('./src', './dist'); \
console.log('Manually generated pluginImports.js'); \
"

# CRITICAL: Add nerve-assessment imports to loadModule function
RUN node -e " \
const fs = require('fs'); \
const filePath = '/app/platform/app/src/pluginImports.js'; \
let content = fs.readFileSync(filePath, 'utf8'); \
\
/* Check if import statement already added (not just push statement) */ \
if (content.includes('import(\"@ohif/extension-nerve-assessment\")')) { \
  console.log('nerve-assessment import already in pluginImports.js'); \
  process.exit(0); \
} \
\
/* Find the last if statement in loadModule and add after it */ \
const extImport = '  if( module===\"@ohif/extension-nerve-assessment\") {\\n    const imported = await import(\"@ohif/extension-nerve-assessment\");\\n    return imported.default;\\n  }\\n'; \
const modeImport = '  if( module===\"@ohif/mode-nerve-assessment\") {\\n    const imported = await import(\"@ohif/mode-nerve-assessment\");\\n    return imported.default;\\n  }\\n'; \
\
/* Find loadModule function and add imports before the closing */ \
const loadModuleMatch = content.match(/async function loadModule\\(module\\) \\{[\\s\\S]*?\\n\\}/); \
if (loadModuleMatch) { \
  const original = loadModuleMatch[0]; \
  const insertPoint = original.lastIndexOf('\\n}'); \
  const modified = original.slice(0, insertPoint) + '\\n' + extImport + modeImport + original.slice(insertPoint); \
  content = content.replace(original, modified); \
  fs.writeFileSync(filePath, content); \
  console.log('SUCCESS: Added nerve-assessment imports to loadModule function'); \
} else { \
  console.log('ERROR: Could not find loadModule function'); \
  process.exit(1); \
} \
"

# Debug: Check pluginImports.js
RUN echo "=== pluginImports.js extensions.push ===" && \
    grep "extensions.push" /app/platform/app/src/pluginImports.js

RUN echo "=== pluginImports.js modes.push ===" && \
    grep "modes.push" /app/platform/app/src/pluginImports.js

# CRITICAL: Modify webpack.pwa.js to NOT overwrite our pluginImports.js
RUN sed -i "s/const copyPluginFromExtensions = writePluginImportFile(SRC_DIR, DIST_DIR);/const copyPluginFromExtensions = []; \/\/ DISABLED/" /app/platform/app/.webpack/webpack.pwa.js && \
    echo "Modified webpack.pwa.js"

# Verify the modification worked
RUN echo "=== webpack.pwa.js after modification ===" && \
    grep -n "copyPluginFromExtensions" /app/platform/app/.webpack/webpack.pwa.js

# CRITICAL: Patch WorkList.tsx to always enable nerve-assessment mode button
# 1. Patch disabled condition
RUN sed -i 's/disabled={!isValidMode}/disabled={!isValidMode \&\& !mode.routeName?.includes("nerve-assessment")}/' /app/platform/app/src/routes/WorkList/WorkList.tsx && \
    echo "Patched WorkList.tsx disabled for nerve-assessment mode"

# 2. Patch onClick preventDefault condition  
RUN sed -i 's/if (!isValidMode) {/if (!isValidMode \&\& !mode.routeName?.includes("nerve-assessment")) {/' /app/platform/app/src/routes/WorkList/WorkList.tsx && \
    echo "Patched WorkList.tsx onClick for nerve-assessment mode"

# Verify WorkList.tsx patches
RUN echo "=== WorkList.tsx patches check ===" && \
    grep -n "disabled=\|if (!isValidMode" /app/platform/app/src/routes/WorkList/WorkList.tsx | head -10

# Verify WorkList.tsx patch
RUN echo "=== WorkList.tsx disabled check ===" && \
    grep -n "disabled=" /app/platform/app/src/routes/WorkList/WorkList.tsx | head -5

# CRITICAL: Patch updateSegmentationStats.ts — NIfTI volume has no DICOM imageIds
# getStatistics() → MetadataProvider::Empty imageId → Error Boundary crash
# Fix: wrap getStatistics() call in try-catch
RUN node -e " \
const fs = require('fs'); \
const fp = '/app/extensions/cornerstone/src/utils/updateSegmentationStats.ts'; \
let c = fs.readFileSync(fp, 'utf8'); \
if (c.includes('statsError')) { console.log('already patched'); process.exit(0); } \
const re = /const stats = await cornerstoneTools\\.utilities\\.segmentation\\.getStatistics\\(\\{[\\s\\S]*?mode:\\s*'individual',[\\s\\S]*?\\}\\);/; \
const m = c.match(re); \
if (m) { \
  c = c.replace(m[0], 'let stats;\\n  try {\\n    stats = await cornerstoneTools.utilities.segmentation.getStatistics({\\n      segmentationId,\\n      segmentIndices,\\n      mode: \\'individual\\',\\n    });\\n  } catch (statsError: any) {\\n    console.warn(\\'[updateSegmentationStats] Stats skipped (non-DICOM volume?):\\', statsError?.message || statsError);\\n    return null;\\n  }'); \
  fs.writeFileSync(fp, c); \
  console.log('SUCCESS: patched updateSegmentationStats.ts'); \
} else { \
  console.log('WARNING: getStatistics pattern not found'); \
  const idx = c.indexOf('getStatistics'); \
  if (idx >= 0) console.log('Context:', c.substring(Math.max(0,idx-80), idx+150)); \
} \
"

# Verify patch
RUN grep -n "statsError" /app/extensions/cornerstone/src/utils/updateSegmentationStats.ts && \
    echo "updateSegmentationStats patch verified" || echo "WARNING: patch not applied"

# Debug: dump cache API keys at build time for verification
RUN node -e " \
console.log('=== Checking cornerstone cache API ==='); \
try { \
  const cs = require('/app/node_modules/@cornerstonejs/core'); \
  const cacheKeys = Object.getOwnPropertyNames(Object.getPrototypeOf(cs.cache)); \
  console.log('cache methods:', cacheKeys.filter(k => k.includes('eometry') || k.includes('olume')).join(', ')); \
} catch(e) { console.log('Could not inspect cache:', e.message); } \
"

# Build with cache disabled
RUN QUICK_BUILD=true NX_SKIP_NX_CACHE=true yarn run build

# Debug: Check generated pluginImports.js after build
RUN echo "=== Generated pluginImports.js ===" && \
    cat /app/platform/app/src/pluginImports.js || echo "Not found"

# Production image with nginx
FROM nginx:alpine

# Copy built OHIF files
COPY --from=builder /app/platform/app/dist /usr/share/nginx/html

# Copy nginx configuration
COPY nginx.conf /etc/nginx/conf.d/default.conf

EXPOSE 80

CMD ["nginx", "-g", "daemon off;"]
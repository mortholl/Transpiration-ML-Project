"""
Study location map for the paper, one point per location (not per site), coloured by biome.

Run it from the QGIS Python console (Plugins > Python Console > Show Editor > open this file > Run),
in a new, empty project, since it clears the current one. It reads study_locations.geojson from the
same folder and writes three files next to it:
    location_map.qgz    the QGIS project, with a print layout called 'Study locations'
    location_map.png    300 dpi export of that layout
    location_map.pdf    vector export of that layout
Everything it builds can be edited by hand afterwards in the layout designer.
"""
import os
from qgis.core import (
    QgsApplication, QgsCategorizedSymbolRenderer, QgsCoordinateReferenceSystem, QgsCoordinateTransform,
    QgsFillSymbol, QgsLayoutExporter, QgsLayoutItem, QgsLayoutItemLabel, QgsLayoutItemLegend,
    QgsLayoutItemMap, QgsLayoutItemPicture, QgsLayoutItemScaleBar, QgsLayoutPoint, QgsLayoutSize,
    QgsLegendRenderer, QgsLegendStyle, QgsMarkerSymbol, QgsPrintLayout, QgsProject, QgsRasterLayer,
    QgsRectangle, QgsRendererCategory, QgsTextFormat, QgsUnitTypes, QgsVectorLayer,
)
from qgis.PyQt.QtCore import Qt
from qgis.PyQt.QtGui import QColor, QFont

# ---- settings -------------------------------------------------------------------------------------

# The folder holding study_locations.geojson, where the outputs are written
FOLDER = r'C:\Users\thorn\Documents\GitHub\Transpiration-ML-Project\paper_visualizations'

# 'esri'    Esri World Topographic tiles, closest to the original figure, needs internet
# 'builtin' the world map that ships with QGIS, offline
# a path    any land or country polygon file, for example Natural Earth's ne_50m_land.shp
BASEMAP = 'esri'

# A scale bar is only true along one latitude on a world Mercator map, so it is off by default
SCALE_BAR = False

# Biome colours, matched to the original figure, in legend order
BIOME_COLOURS = {
    'Temperate forest': '#5b1045',
    'Temperate grassland desert': '#e6259a',
    'Tropical forest savanna': '#4b6584',
    'Tropical rain forest': '#f2e205',
    'Woodland/Shrubland': '#6bcf2e',
}

MAP_WIDTH = 260          # mm, the page is sized around the map
EXTENT = (-180, -58, 180, 80)   # lon/lat window, drops Antarctica and the far Arctic

def text_format(size, colour='#222222'):   # font settings as QGIS 3.30 and later expect them
    fmt = QgsTextFormat()
    fmt.setFont(QFont('Arial'))
    fmt.setSize(size)
    fmt.setSizeUnit(QgsUnitTypes.RenderPoints)
    fmt.setColor(QColor(colour))
    return fmt


# ---- layers ---------------------------------------------------------------------------------------

mm = QgsUnitTypes.LayoutMillimeters
project = QgsProject.instance()
project.clear()
mercator = QgsCoordinateReferenceSystem('EPSG:3857')
project.setCrs(mercator)

if BASEMAP == 'esri':
    base = QgsRasterLayer(
        'type=xyz&zmin=0&zmax=19&url=https://server.arcgisonline.com/ArcGIS/rest/services/'
        'World_Topo_Map/MapServer/tile/%7Bz%7D/%7By%7D/%7Bx%7D', 'Esri World Topographic', 'wms')
    credit = 'Basemap: Esri World Topographic Map'
else:
    path = (os.path.join(QgsApplication.pkgDataPath(), 'resources', 'data', 'world_map.gpkg')
            if BASEMAP == 'builtin' else BASEMAP)
    base = QgsVectorLayer(path, 'Land', 'ogr')
    if base.isValid():
        base.renderer().setSymbol(QgsFillSymbol.createSimple(
            {'color': '#eae8df', 'outline_color': '#b8b6ad', 'outline_width': '0.1'}))
    credit = 'Made with Natural Earth'
if not base.isValid():
    raise RuntimeError(f'Basemap {BASEMAP!r} did not load, check the path or the internet connection')
project.addMapLayer(base)

points = QgsVectorLayer(os.path.join(FOLDER, 'study_locations.geojson'), 'Study locations', 'ogr')
if not points.isValid():
    raise RuntimeError('study_locations.geojson did not load, check FOLDER')
counts = {}
for feature in points.getFeatures():
    counts[feature['biome']] = counts.get(feature['biome'], 0) + 1
unknown = set(counts) - set(BIOME_COLOURS)
if unknown:
    raise RuntimeError(f'No colour set for {sorted(unknown)}, add them to BIOME_COLOURS')

categories = []
for biome, colour in BIOME_COLOURS.items():
    if biome in counts:   # a biome with no locations is left out of the legend
        symbol = QgsMarkerSymbol.createSimple({'name': 'circle', 'color': colour, 'size': '3.2',
                                               'outline_color': '#222222', 'outline_width': '0.3'})
        label = f'{biome} ({counts[biome]})'   # the count is locations, not sites
        categories.append(QgsRendererCategory(biome, symbol, label))
points.setRenderer(QgsCategorizedSymbolRenderer('biome', categories))
project.addMapLayer(points)

# ---- layout ---------------------------------------------------------------------------------------

to_mercator = QgsCoordinateTransform(QgsCoordinateReferenceSystem('EPSG:4326'), mercator, project)
extent = to_mercator.transformBoundingBox(QgsRectangle(*EXTENT))
map_height = MAP_WIDTH * extent.height() / extent.width()

layout = QgsPrintLayout(project)
layout.initializeDefaults()
layout.setName('Study locations')
layout.pageCollection().page(0).setPageSize(QgsLayoutSize(MAP_WIDTH, map_height, mm))
project.layoutManager().addLayout(layout)

map_item = QgsLayoutItemMap(layout)
map_item.attemptMove(QgsLayoutPoint(0, 0, mm))
map_item.attemptResize(QgsLayoutSize(MAP_WIDTH, map_height, mm))
map_item.setCrs(mercator)
map_item.setLayers([points, base])
map_item.setExtent(extent)
map_item.setBackgroundColor(QColor('#cfe3ef'))   # ocean, visible behind a polygon basemap
map_item.setFrameEnabled(True)
layout.addLayoutItem(map_item)

legend = QgsLayoutItemLegend(layout)
legend.setLinkedMap(map_item)
legend.setAutoUpdateModel(False)
root = legend.model().rootGroup()
root.removeLayer(base)
QgsLegendRenderer.setNodeLegendStyle(root.findLayer(points.id()), QgsLegendStyle.Hidden)
legend.setTitle('')
if hasattr(QgsLegendStyle, 'setTextFormat'):
    legend.rstyle(QgsLegendStyle.SymbolLabel).setTextFormat(text_format(8))
else:   # before QGIS 3.30
    legend.setStyleFont(QgsLegendStyle.SymbolLabel, QFont('Arial', 8))
legend.setFrameEnabled(True)
legend.setBackgroundEnabled(True)
legend.setReferencePoint(QgsLayoutItem.LowerLeft)
layout.addLayoutItem(legend)
legend.adjustBoxSize()
legend.attemptMove(QgsLayoutPoint(4, map_height - 4, mm))   # the South Pacific, where there are no locations

label = QgsLayoutItemLabel(layout)
label.setText(credit)
if hasattr(label, 'setTextFormat'):
    label.setTextFormat(text_format(6, '#4a4a4a'))
else:   # before QGIS 3.30
    label.setFont(QFont('Arial', 6))
    label.setFontColor(QColor('#4a4a4a'))
label.attemptResize(QgsLayoutSize(70, 4, mm))
label.setReferencePoint(QgsLayoutItem.LowerRight)
label.attemptMove(QgsLayoutPoint(MAP_WIDTH - 2, map_height - 1, mm))
label.setHAlign(Qt.AlignRight)   # so the credit sits in the bottom right corner
layout.addLayoutItem(label)

arrow_svg = next((os.path.join(folder, 'arrows', 'NorthArrow_02.svg')
                  for folder in QgsApplication.svgPaths()
                  if os.path.exists(os.path.join(folder, 'arrows', 'NorthArrow_02.svg'))), None)
if arrow_svg:
    arrow = QgsLayoutItemPicture(layout)
    arrow.setPicturePath(arrow_svg)
    arrow.attemptResize(QgsLayoutSize(7, 10, mm))
    arrow.setReferencePoint(QgsLayoutItem.LowerRight)
    arrow.attemptMove(QgsLayoutPoint(MAP_WIDTH - 4, map_height - 6, mm))   # above the credit, south of New Zealand
    layout.addLayoutItem(arrow)
else:
    print('No north arrow SVG found, add one from the layout designer if you want it')

if SCALE_BAR:
    bar = QgsLayoutItemScaleBar(layout)
    bar.setLinkedMap(map_item)
    bar.setStyle('Single Box')
    bar.setUnits(QgsUnitTypes.DistanceMiles)
    bar.setUnitLabel('Miles')
    bar.setNumberOfSegments(1)
    bar.setNumberOfSegmentsLeft(0)
    bar.setUnitsPerSegment(2000)
    bar.setHeight(1.5)
    if hasattr(bar, 'setTextFormat'):
        bar.setTextFormat(text_format(7))
    else:
        bar.setFont(QFont('Arial', 7))
    layout.addLayoutItem(bar)
    bar.refresh()
    bar.setReferencePoint(QgsLayoutItem.LowerRight)
    bar.attemptMove(QgsLayoutPoint(MAP_WIDTH - 16, map_height - 6, mm))   # south of Australia, left of the arrow

# ---- outputs --------------------------------------------------------------------------------------

exporter = QgsLayoutExporter(layout)
image_settings = QgsLayoutExporter.ImageExportSettings()
image_settings.dpi = 300
png = os.path.join(FOLDER, 'location_map.png')
pdf = os.path.join(FOLDER, 'location_map.pdf')
if exporter.exportToImage(png, image_settings) != QgsLayoutExporter.Success:
    print('PNG export failed')
if exporter.exportToPdf(pdf, QgsLayoutExporter.PdfExportSettings()) != QgsLayoutExporter.Success:
    print('PDF export failed')
project.write(os.path.join(FOLDER, 'location_map.qgz'))
print(f'{sum(counts.values())} locations across {len(counts)} biomes, wrote location_map.qgz, .png and .pdf')

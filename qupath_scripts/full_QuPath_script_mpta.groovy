// Copyright (C) 2021  Blue Brain Project, EPFL
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <https://www.gnu.org/licenses/>.


import qupath.ext.biop.cellpose.Cellpose2D

def logger = LoggerFactory.getLogger(this.class);

def modelPath =  "/arch/cellpose_model/cellpose_residual_on_style_on_concatenation_off_train_2022_01_11_16_14_20.764792"
def saveFolderPath =  "/arch/Results"
def CountourFinderPath = "/arch/qupath_scripts/CountourFinder.json"
def LayerClassifierPath = "/arch/qupath_scripts/Layer Classiffier.json"

logger.info("cellpose model path: {}", modelPath)
logger.info("Save folder: {}", saveFolderPath)
def saveFolder = new File(saveFolderPath)


def entry = getProjectEntry()
def entryMetadata = entry.getMetadataMap()

def rainbowPower(int n) {
    return (1..n).collect{ i -> Color.HSBtoRGB( (1.0/n)*i, 1.0f, 1.0f) }
}

if (entryMetadata['Analyze'] == 'True') {
    println ("ANALYSE == true")

    // Expand MPtA Annotation
    selectObjectsByPathClass( getPathClass( "MPtA" ) )
    runPlugin('qupath.lib.plugins.objects.DilateAnnotationPlugin', '{"radiusMicrons": 200.0,  "lineCap": "Round",  "removeInterior": false,  "constrainToParent": false}');

    // Pick up the dilated annotation, which is the one with the same class but largest area
    def s1hl = getAnnotationObjects().findAll{ it.getPathClass() == getPathClass("MPtA") }.max{ it.getROI().getArea() }

    // Pick up the SliceContour to subtract it from the dilated MPtA
    def sliceC = getAnnotationObjects().find{ it.getPathClass() == getPathClass( "SliceContour" ) }

    // Do some sexy geometry
    def outPiaROI = RoiTools.combineROIs( s1hl.getROI(), sliceC.getROI(), RoiTools.CombineOp.SUBTRACT )

    // Create the Outside Pia Annotation
    def outPia = PathObjects.createAnnotationObject(outPiaROI, getPathClass( "Outside Pia" ) )


    addObject( outPia )

    // Delete original dilated MPtA
    removeObject(s1hl, true)

    // Cleanup and show
    resetSelection()
    fireHierarchyUpdate()


    // Create needed annotation for the Object Classifier
    def layers = [ "Layer 1",
                   "Layer 2",
                   "Layer 3",
                   "Layer 4",
                   "Layer 5",
                   "Layer 6 a",
                   "Layer 6 b",
                   "Outside Pia",
                   "MPtA",
                   "SliceContour"
                  ]
    println ("layer.size()")
    println(layers.size() )

    def colors = rainbowPower( layers.size() )


    def qupath = getQuPath()
    if (qupath != null) {

        def available = qupath.getAvailablePathClasses()


        def pathClasses = [layers, colors].transpose().collect{ name, color ->
            def newClass = PathClassFactory.getPathClass(name)
            newClass.setColor(color)
            return newClass
        }
        Platform.runLater{
            qupath.resetAvailablePathClasses()
            available.addAll( pathClasses )
        }
     }



    println 'Add annotations Done !'


    // Model trained on full size image, with cyto2 model as base for 1200 epochs
    def cellpose = Cellpose2D.builder(modelPath)
            .pixelSize(0.3460)
            .tileSize(2048)
            .diameter(30)                // Average diameter of objects in px (at the requested pixel sie)
            .measureShape()              // Add shape measurements
            .measureIntensity()          // Add cell measurements (in all compartments)
            .classify("Cellpose Julie Full")
            .build()

    println "cellpose $cellpose"

    // Run detection for the selected objects
    def imageData = getCurrentImageData()

    def pathObjects = getObjects{ it.getPathClass().equals( getPathClass( "MPtA" ) ) }
    //if (pathObjects.isEmpty()) {
    //    Dialogs.showErrorMessage("Cellpose", "Please select a parent object!")
    //}
    println "Execute cellpose on $pathObjects"
    println "imageData $imageData"
    cellpose.detectObjects(imageData, pathObjects)

    println 'Cellpose algorithm for cellular segmentation Done!'
    // CIRCULARITY CORRECTION START

    // Run this code after cellpose

    def cells = getDetectionObjects()

    def multis = cells.findAll{ it.getROI().getGeometry() instanceof org.locationtech.jts.geom.MultiPolygon }

    // fix
    def fixedMultis = multis.collect{ cell ->
        def geom = cell.getROI().getGeometry()
        // Get the largest geometry and assume that it is the cell
        def idx = (0..< geom.getNumGeometries()).max{ geom.getGeometryN( it ).getArea() }

        def newROI = GeometryTools.geometryToROI( geom.getGeometryN( idx ), null )
        def newcell = PathObjects.createDetectionObject( newROI, cell.getPathClass(), cell.getMeasurementList() )
    }

    // Add what is needed and remove the old ones
    removeObjects( multis, false )
    addObjects( fixedMultis )
    fireHierarchyUpdate()

    // Re-compute measurements
    selectDetections()
    addShapeMeasurements("AREA", "LENGTH", "CIRCULARITY", "SOLIDITY", "MAX_DIAMETER", "MIN_DIAMETER", "NUCLEUS_CELL_RATIO")
    // CIRCULARY CORRECTION END


    // Add features for classifer and run it
    detectionToAnnotationDistances(true)
    selectAnnotations()
    runPlugin('qupath.opencv.features.DelaunayClusteringPlugin', '{"distanceThresholdMicrons": 0.0,  "limitByClass": false,  "addClusterMeasurements": false}')
    runPlugin('qupath.lib.plugins.objects.SmoothFeaturesPlugin', '{"fwhmMicrons": 50.0,  "smoothWithinClasses": false}')
    //runObjectClassifier("Layer Classiffier")
    //runObjectClassifier(LayerClassifierPath)
    println 'Add features for classifer and run it Done!'


    //getDetectionObjects().each{ it.setPathClass( getPathClass("CellPose") ) }

    // 1. Save Detection Measurements, keeping useful lines from `save_detection_measurement.groovy`
    setImageType('BRIGHTFIELD_OTHER');
    setColorDeconvolutionStains('{"Name" : "H-DAB default", "Stain 1" : "Hematoxylin", "Values 1" : "0.65111 0.70119 0.29049", "Stain 2" : "DAB", "Values 2" : "0.26917 0.56824 0.77759", "Background" : " 255 255 255"}');
    resetSelection();
    createAnnotationsFromPixelClassifier(CountourFinderPath, 1000000.0, 1000000.0)
    runPlugin('qupath.lib.plugins.objects.SplitAnnotationsPlugin', '{}')
    saveDetectionMeasurements( saveFolder.getAbsolutePath() )

    // 2. Save annotations to folder from 'export_annotations_for_pipeline'
    def annotations = getAnnotationObjects()
    def writer = new WKTWriter()
    def gson = GsonTools.getInstance(true)
    // Pick up current image name to append to the resulting files
    def imageName = getCurrentServer().getMetadata().getName()

    def file = new File(saveFolder,imageName + '_annotations.json' )
    annotations.each {
        println writer.write( it.getROI().getGeometry() )
        file.withWriter('UTF-8') {
            gson.toJson( annotations,it )
        }
    }

    println('Export and save Detection Measurements and annotations Done')


}
else {
        println ("ANALYSE == false")
}
// Imports
import qupath.lib.objects.classes.PathClassFactory
import org.locationtech.jts.io.WKTWriter
import org.slf4j.LoggerFactory;
import java.awt.Color;

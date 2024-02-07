#!/usr/bin/env python
# ******************************************************************************
# Copyright (c) 2021 Bentley Systems, Incorporated. All rights reserved.
# ******************************************************************************
# ContextCapture MasterKernel Python SDK - example script
#
# Script: context_insights_sample.py
# Purpose : Submit standalone context insights jobs, on premise and cloud
# Keywords: insights upload, insights download, insights job submission
#
# Script description:
# - Upload reality data, keeping track of reality data ids to avoid re-uploading
# - Submit several jobs
# - Monitor jobs and download results of cloud jobs
# ******************************************************************************

import os
from transfer_data import UploadEntry, upload_data, DataType
from job_manager import JobManager, PCType
import argparse
import ccmasterkernel

SUCCESS_CODE = 0
ERROR_CODE = 1

parser = argparse.ArgumentParser()
parser.add_argument('--nocloud', dest='nocloud', action='store_true', default=False, help="If True, skip RDAS cloud jobs.")
parser.add_argument('--noprem', dest='noprem', action='store_true', default=False, help="If True, skip on premise jobs.")
parser.add_argument('--output', dest='output', required=True, help="Output directory")
args = parser.parse_args()

output_path = args.output                                       # Where results should be saved
nocloud = args.nocloud                                          # Skip cloud jobs
noprem = args.noprem                                            # Skip in premise jobs

input_path = r"Q:\Analyze\ContextInsightsQA_デテクター"           # Where input data is stored

# rdas_url = "https://dev-connect-contextinsights.bentley.com/api/v1"
rdas_url = "https://qa-api.bentley.com/realitydataanalysis"
# rdas_url = "https://dev-api.bentley.com/realitydataanalysis"
itwin_id = "ad14b27c-91ea-4492-9433-1e2d6903b5e4"             # RDS iTwin id
references_path = os.path.join(input_path, "references-qa.txt")    # Where reality data ids for data below should be saved

# rdas_url = "https://connect-contextinsights.bentley.com/api/v1"
# rdas_url = "https://api.bentley.com/realitydataanalysis"
# itwin_id = "d371d643-a3ac-49b0-ac3f-111d74d52b3c"                 # RDS iTwin id
# references_path = os.path.join(input_path, "references-prod.txt")   # Where reality data ids for data below should be saved

if not nocloud:
    # ============================ Configure cloud service ======================

    if not ccmasterkernel.InsightStandaloneJob.configureService(rdas_url=rdas_url).isNone():
        print("Failed to configure cloud service")
        exit(ERROR_CODE)

    # ============================ Upload data ======================
    # Data to upload to RDS
    data_to_upload = [
        # Detectors
        UploadEntry("Coco photo object detector",
                    os.path.join(input_path, r"Detectors\PhotoObjectDetector\Coco2017_v1"),
                    DataType.ContextDetector),
        UploadEntry("Coco low confidence photo object detector",
                    os.path.join(input_path, r"Detectors\PhotoObjectDetector\Coco2017_low_confidence_v1"),
                    DataType.ContextDetector),
        UploadEntry("Pascal photo segmentation detector",
                    os.path.join(input_path, r"Detectors\PhotoSegmentationDetector\PascalVoc2012_v1"),
                    DataType.ContextDetector),
        UploadEntry("RoofB_v3 orthophoto segmentation detector",
                    os.path.join(input_path, r"Detectors\OrthophotoSegmentationDetector\RGB\RoofB_v3"),
                    DataType.ContextDetector),
        UploadEntry("CityA_RGBD_v1 orthophoto segmentation detector",
                    os.path.join(input_path,
                                 r"Detectors\OrthophotoSegmentationDetector\RGB_DSM\CityA_RGBD_v1"),
                    DataType.ContextDetector),
        UploadEntry("CracksAppliedAI_v1 photo segmentation detector",
                    os.path.join(input_path, r"Detectors\PhotoSegmentationDetector\CracksAppliedAI_v1"),
                    DataType.ContextDetector),
        UploadEntry("CracksAppliedAI_v1 orthophoto segmentation detector",
                    os.path.join(input_path, r"Detectors\OrthophotoSegmentationDetector\RGB\CracksAppliedAI_v1"),
                    DataType.ContextDetector),
        UploadEntry("CracksA_v1 orthophoto segmentation detector",
                    os.path.join(input_path, r"Detectors\OrthophotoSegmentationDetector\RGB\CracksA_v1"),
                    DataType.ContextDetector),
        UploadEntry("CracksA_v1 photo segmentation detector",
                    os.path.join(input_path, r"Detectors\PhotoSegmentationDetector\CracksA_v1"),
                    DataType.ContextDetector),
        UploadEntry("RailA_v5 photo object detector",
                    os.path.join(input_path, r"Detectors\PhotoObjectDetector\RailA_v5"),
                    DataType.ContextDetector),
        UploadEntry("Antennas photo object detector",
                    os.path.join(input_path, r"Detectors\PhotoObjectDetector\Antennas_v1"),
                    DataType.ContextDetector),
        UploadEntry("Dales point cloud segmentation detector",
                    os.path.join(input_path, r"Detectors\PointCloudSegmentationDetector\Dales_v2"),
                    DataType.ContextDetector),
        UploadEntry("Hessigheim point cloud segmentation detector",
                    os.path.join(input_path, r"Detectors\PointCloudSegmentationDetector\Hessigheim_v3"),
                    DataType.ContextDetector),
        UploadEntry("Rail point cloud segmentation detector",
                    os.path.join(input_path, r"Detectors\PointCloudSegmentationDetector\RailA_v5"),
                    DataType.ContextDetector),
        UploadEntry("Test point cloud segmentation detector",
                    os.path.join(input_path, r"Detectors\PointCloudSegmentationDetector\Test"),
                    DataType.ContextDetector),
        UploadEntry("Wingtra point cloud segmentation detector",
                    os.path.join(input_path, r"Detectors\PointCloudSegmentationDetector\Wingtra_v1"),
                    DataType.ContextDetector),
        UploadEntry("Trees point cloud segmentation detector",
                    os.path.join(input_path, r"Detectors\PointCloudSegmentationDetector\Trees_v1"),
                    DataType.ContextDetector),

        # Scenes and dependencies
        UploadEntry("Motos images",
                    os.path.join(input_path, r"DataSets\Motos\Images_バイク"),
                    DataType.ImageCollection),
        UploadEntry("Motos photos",
                    os.path.join(input_path, r"DataSets\Motos\Photos"),
                    DataType.ContextScene),

        UploadEntry("Christchurch orthotiles",
                    os.path.join(input_path, r"DataSets\Christchurch\Images"),
                    DataType.ImageCollection),
        UploadEntry("Christchurch orthophoto",
                    os.path.join(input_path, r"DataSets\Christchurch\Orthophoto"),
                    DataType.ContextScene),

        UploadEntry("Christchurch JPEG orthotiles",
                    os.path.join(input_path, r"DataSets\Christchurch_JPEG\Images"),
                    DataType.ImageCollection),
        UploadEntry("Christchurch JPEG orthophoto",
                    os.path.join(input_path, r"DataSets\Christchurch_JPEG\Orthophoto"),
                    DataType.ContextScene),

        UploadEntry("GrazOrtho orthotiles",
                    os.path.join(input_path, r"DataSets\Graz\OrthophotoImages"),
                    DataType.ImageCollection),
        UploadEntry("GrazOrtho orthophoto",
                    os.path.join(input_path, r"DataSets\Graz\OrthophotoScene"),
                    DataType.ContextScene),
        UploadEntry("Graz 3MX", os.path.join(input_path, r"DataSets\Graz\3MX"), DataType.ThreeMX),
        UploadEntry("Graz 3MX scene", os.path.join(input_path, r"DataSets\Graz\Mesh3MX"), DataType.ContextScene),
        UploadEntry("Graz 3SM", os.path.join(input_path, r"DataSets\Graz\3SM"), DataType.ThreeSM),
        UploadEntry("Graz 3SM scene", os.path.join(input_path, r"DataSets\Graz\Mesh3SM"), DataType.ContextScene),

        UploadEntry("MinnesotaDOT orthotiles",
                    os.path.join(input_path, r"DataSets\MinnesotaDOT\OrthophotoImages"),
                    DataType.ImageCollection),
        UploadEntry("MinnesotaDOTOrtho orthophoto",
                    os.path.join(input_path, r"DataSets\MinnesotaDOT\OrthophotosScene"),
                    DataType.ContextScene),

        UploadEntry("BridgeWithCracks images",
                    os.path.join(input_path, r"DataSets\BridgeWithCracks\Images"),
                    DataType.ImageCollection),
        UploadEntry("BridgeWithCracks oriented photos",
                    os.path.join(input_path, r"DataSets\BridgeWithCracks\OrientedPhotos"),
                    DataType.ContextScene),
        UploadEntry("BridgeWithCracks CCOrientations QA",
                    os.path.join(input_path, r"DataSets\BridgeWithCracks\OrientationsWorkAroundQA"),
                    DataType.CCOrientations),
        UploadEntry("BridgeWithCracks 3MX", os.path.join(input_path, r"DataSets\BridgeWithCracks\3MX"),
                    DataType.ThreeMX),
        UploadEntry("BridgeWithCracks 3MX scene", os.path.join(input_path, r"DataSets\BridgeWithCracks\Mesh3MX"),
                    DataType.ContextScene),
        UploadEntry("BridgeWithCracks 3SM", os.path.join(input_path, r"DataSets\BridgeWithCracks\3SM"),
                    DataType.ThreeSM),
        UploadEntry("BridgeWithCracks 3SM scene", os.path.join(input_path, r"DataSets\BridgeWithCracks\Mesh3SM"),
                    DataType.ContextScene),
        UploadEntry("BridgeWithCracks Segmentation2D",
                    os.path.join(input_path, r"GivenDetections\S2D_BridgeWithCracks"), DataType.ContextScene),
        UploadEntry("BridgeWithCracks POD", os.path.join(input_path, r"DataSets\BridgeWithCracks\POD"),
                    DataType.POD),
        UploadEntry("BridgeWithCracks POD scene",
                    os.path.join(input_path, r"DataSets\BridgeWithCracks\PointClouds"),
                    DataType.ContextScene),
        UploadEntry("BridgeWithCracks OPC", os.path.join(input_path, r"DataSets\BridgeWithCracks\OPC"),
                    DataType.OPC),
        UploadEntry("BridgeWithCracks OPC scene",
                    os.path.join(input_path, r"DataSets\BridgeWithCracks\PointClouds_OPC"),
                    DataType.ContextScene),

        UploadEntry("Spain images 2",
                    os.path.join(input_path, r"DataSets\Spain\Images\planar2\original"),
                    DataType.ImageCollection),
        UploadEntry("Spain images 3",
                    os.path.join(input_path, r"DataSets\Spain\Images\planar3\original"),
                    DataType.ImageCollection),
        UploadEntry("Spain images 4",
                    os.path.join(input_path, r"DataSets\Spain\Images\planar4\original"),
                    DataType.ImageCollection),
        UploadEntry("Spain images 5",
                    os.path.join(input_path, r"DataSets\Spain\Images\planar5\original"),
                    DataType.ImageCollection),
        UploadEntry("Spain images 6",
                    os.path.join(input_path, r"DataSets\Spain\Images\planar6\original"),
                    DataType.ImageCollection),
        UploadEntry("Spain oriented photos",
                    os.path.join(input_path, r"DataSets\Spain\OrientedPhotos"),
                    DataType.ContextScene),
        UploadEntry("Spain Objects2D",
                    os.path.join(input_path, r"GivenDetections\O2D_Spain"),
                    DataType.ContextScene),
        UploadEntry("Spain POD",
                    os.path.join(input_path, r"DataSets\Spain\POD"),
                    DataType.POD),
        UploadEntry("Spain POD scene",
                    os.path.join(input_path, r"DataSets\Spain\PointCloudsPOD"),
                    DataType.ContextScene),
        UploadEntry("Spain POD scene",
                    os.path.join(input_path, r"DataSets\Spain\PointCloudsNoGeo"),
                    DataType.ContextScene),
        UploadEntry("Spain OPC",
                    os.path.join(input_path, r"DataSets\Spain\OPC"),
                    DataType.OPC),
        UploadEntry("Spain OPC scene",
                    os.path.join(input_path, r"DataSets\Spain\PointCloudsOPC"),
                    DataType.ContextScene),
        UploadEntry("Spain LAS", os.path.join(input_path, r"DataSets\Spain\LAS"), DataType.LAS),
        UploadEntry("Spain LAS scene", os.path.join(input_path, r"DataSets\Spain\PointCloudsLAS"),
                    DataType.ContextScene),
        UploadEntry("Spain Segmentation3D v3", os.path.join(input_path, r"GivenDetections\S3D_Spain"),
                    DataType.ContextScene),
        UploadEntry("Spain Segmentation3D OPC",
                    os.path.join(input_path, r"GivenDetections\S3D_Spain_OPC\segmentedPointCloud"), DataType.OPC),
        UploadEntry("Spain Segmentation3D", os.path.join(input_path, r"GivenDetections\S3D_Spain_OPC\segmentation3D"),
                    DataType.ContextScene),
        UploadEntry("Spain Segmentation3D NoGeo OPC",
                    os.path.join(input_path, r"GivenDetections\S3D_SpainNoGeo\segmentedPointCloud"), DataType.OPC),
        UploadEntry("Spain Segmentation3D NoGeo", os.path.join(input_path, r"GivenDetections\S3D_SpainNoGeo\segmentation3D"),
                    DataType.ContextScene),

        UploadEntry("Spain 1 POD", os.path.join(input_path, r"DataSets\SpainChange\POD_before"), DataType.POD),
        UploadEntry("Spain 2 POD", os.path.join(input_path, r"DataSets\SpainChange\POD_after"), DataType.POD),
        UploadEntry("Spain 1 POD scene", os.path.join(input_path, r"DataSets\SpainChange\PointClouds1"),
                    DataType.ContextScene),
        UploadEntry("Spain 2 POD scene", os.path.join(input_path, r"DataSets\SpainChange\PointClouds2"),
                    DataType.ContextScene),

        UploadEntry("Antennas images",
                    os.path.join(input_path, r"DataSets\Antennas\Images"),
                    DataType.ImageCollection),
        UploadEntry("Antennas oriented photos",
                    os.path.join(input_path, r"DataSets\Antennas\OrientedPhotos"),
                    DataType.ContextScene),
        UploadEntry("Antennas Objects2D",
                    os.path.join(input_path, r"GivenDetections\O2D_Antennas"),
                    DataType.ContextScene),

        UploadEntry("Dales POD", os.path.join(input_path, r"DataSets\Dales\POD"), DataType.POD),
        UploadEntry("Dales PointCloud", os.path.join(input_path, r"DataSets\Dales\Pointclouds"), DataType.ContextScene),
        UploadEntry("Dales Segmentation3D v3", os.path.join(input_path, r"GivenDetections\S3D_Dales"),
                    DataType.ContextScene),
        UploadEntry("Dales Segmentation3D v3 with confidence",
                    os.path.join(input_path, r"GivenDetections\S3D_Dales_with_Confidence"), DataType.ContextScene),
        UploadEntry("Dales Segmentation3D OPC",
                    os.path.join(input_path, r"GivenDetections\S3D_Dales_OPC\segmentedPointCloud"), DataType.OPC),
        UploadEntry("Dales Segmentation3D", os.path.join(input_path, r"GivenDetections\S3D_Dales_OPC\segmentation3D"),
                    DataType.ContextScene),

        UploadEntry("Tuxford LAZ", os.path.join(input_path, r"DataSets\Tuxford\LAZ"), DataType.LAZ),
        UploadEntry("Tuxford LAZ scene", os.path.join(input_path, r"DataSets\Tuxford\PointCloud"),
                    DataType.ContextScene),

        UploadEntry("Horn 1 LAZ", os.path.join(input_path, r"DataSets\Horn\LAZ\LAZ_before"), DataType.LAZ),
        UploadEntry("Horn 2 LAZ", os.path.join(input_path, r"DataSets\Horn\LAZ\LAZ_after"), DataType.LAZ),
        UploadEntry("Horn 1 LAZ scene", os.path.join(input_path, r"DataSets\Horn\LAZ\PointClouds1"),
                    DataType.ContextScene),
        UploadEntry("Horn 2 LAZ scene", os.path.join(input_path, r"DataSets\Horn\LAZ\PointClouds2"),
                    DataType.ContextScene),

        UploadEntry("Horn 1 OPC", os.path.join(input_path, r"DataSets\Horn\OPC\OPC_before"), DataType.OPC),
        UploadEntry("Horn 2 OPC", os.path.join(input_path, r"DataSets\Horn\OPC\OPC_after"), DataType.OPC),
        UploadEntry("Horn 1 OPC scene", os.path.join(input_path, r"DataSets\Horn\OPC\PointClouds1"),
                    DataType.ContextScene),
        UploadEntry("Horn 2 OPC scene", os.path.join(input_path, r"DataSets\Horn\OPC\PointClouds2"),
                    DataType.ContextScene),

        UploadEntry("Horn 1 NoSRS LAS", os.path.join(input_path, r"DataSets\Horn\NoSRS\NoSRS_before"), DataType.LAS),
        UploadEntry("Horn 2 NoSRS LAS", os.path.join(input_path, r"DataSets\Horn\NoSRS\NoSRS_after"), DataType.LAS),
        UploadEntry("Horn 1 NoSRS scene", os.path.join(input_path, r"DataSets\Horn\NoSRS\PointClouds1"),
                    DataType.ContextScene),
        UploadEntry("Horn 2 NoSRS scene", os.path.join(input_path, r"DataSets\Horn\NoSRS\PointClouds2"),
                    DataType.ContextScene),
        UploadEntry("Horn 1 PLY", os.path.join(input_path, r"DataSets\Horn\PLY\PLY_before"), DataType.PLY),
        UploadEntry("Horn 2 PLY", os.path.join(input_path, r"DataSets\Horn\PLY\PLY_after"), DataType.PLY),
        UploadEntry("Horn 1 PLY scene", os.path.join(input_path, r"DataSets\Horn\PLY\PointClouds1"),
                    DataType.ContextScene),
        UploadEntry("Horn 2 PLY scene", os.path.join(input_path, r"DataSets\Horn\PLY\PointClouds2"),
                    DataType.ContextScene),
        UploadEntry("Horn 1 POD", os.path.join(input_path, r"DataSets\Horn\POD\POD_before"), DataType.POD),
        UploadEntry("Horn 2 POD", os.path.join(input_path, r"DataSets\Horn\POD\POD_after"), DataType.POD),
        UploadEntry("Horn 1 POD scene", os.path.join(input_path, r"DataSets\Horn\POD\PointClouds1"),
                    DataType.ContextScene),
        UploadEntry("Horn 2 POD scene", os.path.join(input_path, r"DataSets\Horn\POD\PointClouds2"),
                    DataType.ContextScene),

        UploadEntry("Lake 1 3MX", os.path.join(input_path, r"DataSets\Lake\WithoutConstraint"), DataType.ThreeMX),
        UploadEntry("Lake 1 3MX scene", os.path.join(input_path, r"DataSets\Lake\Meshes1"), DataType.ContextScene),
        UploadEntry("Lake 2 3MX", os.path.join(input_path, r"DataSets\Lake\WithConstraint"), DataType.ThreeMX),
        UploadEntry("Lake 2 3MX scene", os.path.join(input_path, r"DataSets\Lake\Meshes2"), DataType.ContextScene),

        UploadEntry("ArGodoy extract OPC", os.path.join(input_path, r"DataSets\ArGodoyCruz_Tree\Extract\OPC"), DataType.OPC),
        UploadEntry("ArGodoy extract OPC scene", os.path.join(input_path, r"DataSets\ArGodoyCruz_Tree\Extract\PointCloudsOPC"),DataType.ContextScene),

        UploadEntry("ArGodoy given S3D OPC",
                                os.path.join(input_path, r"GivenDetections\S3D_ArGodoyCruz\segmentedPointCloud"), DataType.OPC),
        UploadEntry("ArGodoy given S3D scene",
                               os.path.join(input_path, r"GivenDetections\S3D_ArGodoyCruz\segmentation3D"),
                                             DataType.ContextScene)
    ]

    if not upload_data(itwin_id, references_path, data_to_upload):
        print("Failed to upload data")
        exit(ERROR_CODE)

# ============================ Init job manager ======================

job_manager = JobManager(references_path)
job_manager.setCallback()

# ============================== Submit on premise jobs ====================
if not noprem:

    # ========================= O2D

    # Detecting objects in photos
    success, job_id = job_manager.addObjects2DJob(job_name="O2D_moto_バイク",
                                                  photos=os.path.join(input_path,
                                                                      r"DataSets\Motos\Photos"),
                                                  photo_object_detector=os.path.join(input_path,
                                                                                     r"Detectors\PhotoObjectDetector\Coco2017_v1"),
                                                  output=os.path.join(output_path, "O2D_motos_バイク")
                                                  )

    # Uncomment for a quick QA test with one job only
    # if qa:
    #     job_manager.start_monitoring()
    #     exit(SUCCESS_CODE)

    success, job_id = job_manager.addObjects2DJob(job_name="O2D_moto_low_confidence_バイク",
                                                  photos=os.path.join(input_path,
                                                                      r"DataSets\Motos\Photos"),
                                                  photo_object_detector=os.path.join(input_path,
                                                                                     r"Detectors\PhotoObjectDetector\Coco2017_low_confidence_v1"),
                                                  output=os.path.join(output_path, "O2D_moto_low_confidence_バイク")
                                                  )

    # ========================= S2D

    # Classifying pixels in photos
    success, job_id = job_manager.addSegmentation2DJob(job_name="S2D_motos_バイク",
                                                       photos=os.path.join(input_path,
                                                                           r"DataSets\Motos\Photos"),
                                                       photo_segmentation_detector=os.path.join(input_path,
                                                                                                r"Detectors\PhotoSegmentationDetector\PascalVoc2012_v1"),
                                                       output=os.path.join(output_path, "S2D_motos_バイク")
                                                       )

    success, job_id = job_manager.addSegmentation2DJob(job_name="S2D_BridgeWithCracks_オルト",
                                                photos=os.path.join(input_path,
                                                                             r"DataSets\BridgeWithCracks\OrientedPhotos"),
                                                photo_segmentation_detector=os.path.join(input_path,
                                                                                         r"Detectors\PhotoSegmentationDetector\CracksA_v1"),
                                                output=os.path.join(output_path, "S2D_BridgeWithCracks_オルト")
                                                )

    # Classifying pixels in photos with AppliedAI Onnx model
    success, job_id = job_manager.addSegmentation2DJob(job_name="S2D_BridgeWithCracks_ONNX_オルト",
                                                photos=os.path.join(input_path,
                                                                             r"DataSets\BridgeWithCracks\OrientedPhotos"),
                                                photo_segmentation_detector=os.path.join(input_path,
                                                                                         r"Detectors\PhotoSegmentationDetector\CracksAppliedAI_v1"),
                                                output=os.path.join(output_path, "S2D_BridgeWithCracks_ONNX_オルト")
                                                )

    # Classifying pixels in orthophoto and exporting 2D polygons
    success, job_id = job_manager.addSegmentation2DJob(job_name="S2D_christchurch_オルト",
                                                       orthophoto=os.path.join(input_path,
                                                                               r"DataSets\Christchurch\Orthophoto"),
                                                       orthophoto_segmentation_detector=os.path.join(input_path,
                                                                                                     r"Detectors\OrthophotoSegmentationDetector\RGB\RoofB_v3"),
                                                       detect_polygons2D=True,
                                                       export_polygons2D=True,
                                                       output=os.path.join(output_path, "S2D_christchurch_オルト"),
                                                       )

    success, job_id = job_manager.addSegmentation2DJob(job_name="S2D_christchurch_JPEG_オルト",
                                                       orthophoto=os.path.join(input_path,
                                                                               r"DataSets\Christchurch_JPEG\Orthophoto"),
                                                       orthophoto_segmentation_detector=os.path.join(input_path,
                                                                                                     r"Detectors\OrthophotoSegmentationDetector\RGB\RoofB_v3"),
                                                       detect_polygons2D=True,
                                                       export_polygons2D=True,
                                                       output=os.path.join(output_path, "S2D_christchurch_JPEG_オルト")
                                                       )

    # Classifying pixels in orthophoto with DSM and exporting 2d polygons
    success, job_id = job_manager.addSegmentation2DJob(job_name="S2D_Graz_オルト",
                                                       orthophoto=os.path.join(input_path,
                                                                               r"DataSets\Graz\OrthophotoScene"),
                                                       orthophoto_segmentation_detector=os.path.join(input_path,
                                                                                                     r"Detectors\OrthophotoSegmentationDetector\RGB_DSM\CityA_RGBD_v1"),
                                                       detect_polygons2D=True,
                                                       export_polygons2D=True,
                                                       output=os.path.join(output_path, "S2D_Graz_オルト")
                                                       )

    # Classifing pixels in orthophoto and detecting 2D lines
    success, job_id = job_manager.addSegmentation2DJob(job_name="S2D_MinnesotaDOT_オルト",
                                                       orthophoto=os.path.join(input_path,
                                                                               r"DataSets\MinnesotaDOT\OrthophotosScene"),
                                                       orthophoto_segmentation_detector=os.path.join(input_path,
                                                                                                     r"Detectors\OrthophotoSegmentationDetector\RGB\CracksA_v1"),
                                                       detect_lines2D=True,
                                                       export_lines2DSHP=True,
                                                       output=os.path.join(output_path, "S2D_MinnesotaDOT_オルト")
                                                       )

    success, job_id = job_manager.addSegmentation2DJob(job_name="S2D_MinnesotaDOT_Onnx_オルト",
                                                       orthophoto=os.path.join(input_path,
                                                                               r"DataSets\MinnesotaDOT\OrthophotosScene"),
                                                       orthophoto_segmentation_detector=os.path.join(input_path,
                                                                                                     r"Detectors\OrthophotoSegmentationDetector\RGB\CracksAppliedAI_v1"),
                                                       detect_lines2D=True,
                                                       export_lines2DDGN=True,
                                                       output=os.path.join(output_path, "S2D_MinnesotaDOT_Onnx_オルト")
                                                       )

    # ========================= O3D

    # Detecting objects in photos and objects in point clouds
    success, job_id = job_manager.addObjects3DJob(job_name="O3D_spain_オルト",
                                                  oriented_photos=os.path.join(input_path,
                                                                               r"DataSets\Spain\OrientedPhotos"),
                                                  point_clouds=os.path.join(input_path,
                                                                            r"DataSets\Spain\PointCloudsPOD"),
                                                  photo_object_detector=os.path.join(input_path,
                                                                                     r"Detectors\PhotoObjectDetector\RailA_v5"),
                                                  export_srs="EPSG:32629",
                                                  export_objects3DDGN=True,
                                                  export_locations3DSHP=True,
                                                  export_objects3DCesium=True,
                                                  output=os.path.join(output_path, "O3D_spain_オルト")
                                                  )

    success, job_id = job_manager.addObjects3DJob(job_name="O3D_spain_OPC_オルト",
                                                  oriented_photos=os.path.join(input_path,
                                                                               r"DataSets\Spain\OrientedPhotos"),
                                                  point_clouds=os.path.join(input_path,
                                                                            r"DataSets\Spain\PointCloudsOPC"),
                                                  photo_object_detector=os.path.join(input_path,
                                                                                     r"Detectors\PhotoObjectDetector\RailA_v5"),
                                                  export_srs="EPSG:32629",
                                                  export_objects3DDGN=True,
                                                  export_locations3DSHP=True,
                                                  export_objects3DCesium=True,
                                                  output=os.path.join(output_path, "O3D_spain_OPC_オルト")
                                                  )

    # Detecting objects in photos and objects in space
    success, job_id = job_manager.addObjects3DJob(job_name="O3D_antennas_オルト",
                                                  oriented_photos=os.path.join(input_path,
                                                                               r"DataSets\Antennas\OrientedPhotos"),
                                                  photo_object_detector=os.path.join(input_path,
                                                                                     r"Detectors\PhotoObjectDetector\Antennas_v1"),
                                                  export_objects3DDGN=True,
                                                  export_objects3DCesium=True,
                                                  export_locations3DSHP=True,
                                                  output=os.path.join(output_path, "O3D_antennas_オルト")
                                                  )

    # Given objects in photos, detecting objects in point clouds
    success, job_id = job_manager.addObjects3DJob(job_name="O3D_spain_given_O2D_オルト",
                                                  oriented_photos=os.path.join(input_path,
                                                                               r"DataSets\Spain\OrientedPhotos"),
                                                  point_clouds=os.path.join(input_path,
                                                                            r"DataSets\Spain\PointCloudsPOD"),
                                                  objects2D=os.path.join(input_path,
                                                                         r"GivenDetections\O2D_Spain"),
                                                  export_srs="EPSG:32629",
                                                  export_objects3DDGN=True,
                                                  export_objects3DCesium=True,
                                                  export_locations3DSHP=True,
                                                  output=os.path.join(output_path, "O3D_spain_given_O2D_オルト")
                                                  )


    success, job_id = job_manager.addObjects3DJob(job_name="O3D_spain_given_O2D_OPC_オルト",
                                                  oriented_photos=os.path.join(input_path,
                                                                               r"DataSets\Spain\OrientedPhotos"),
                                                  point_clouds=os.path.join(input_path,
                                                                            r"DataSets\Spain\PointCloudsOPC"),
                                                  objects2D=os.path.join(input_path,
                                                                         r"GivenDetections\O2D_Spain"),
                                                  export_srs="EPSG:32629",
                                                  export_objects3DDGN=True,
                                                  export_locations3DSHP=True,
                                                  export_objects3DCesium=True,
                                                  output=os.path.join(output_path, "O3D_spain_given_O2D_OPC_オルト")
                                                  )


    # Given objects in photos, detecting objects in space
    success, job_id = job_manager.addObjects3DJob(job_name="O3D_antennas_given_O2D_オルト",
                                                  oriented_photos=os.path.join(input_path,
                                                                               r"DataSets\Antennas\OrientedPhotos"),
                                                  objects2D=os.path.join(input_path,
                                                                         r"GivenDetections\O2D_Antennas"),
                                                  export_objects3DDGN=True,
                                                  export_objects3DCesium=True,
                                                  export_locations3DSHP=True,
                                                  output=os.path.join(output_path, "O3D_antennas_given_O2D_オルト")
                                                  )

    # ========================= S3D

    # Classifying point clouds and detecting objects in space
    success, job_id = job_manager.addSegmentation3DJob(job_name="S3D_dales_オルト",
                                                       point_clouds=os.path.join(input_path,
                                                                                 r"DataSets\Dales\Pointclouds"),
                                                       point_cloud_segmentation_detector=os.path.join(input_path,
                                                                                                      r"Detectors\PointCloudSegmentationDetector\Dales_v2"),
                                                       save_confidence=True,
                                                       export_srs="EPSG:26910",
                                                       export_segmentation3D=[PCType.LAZ, PCType.POD, PCType.PLY],
                                                       detect_objects3D=True,
                                                       export_objects3DDGN=True,
                                                       export_objects3DCesium=True,
                                                       export_locations3DSHP=True,
                                                       output=os.path.join(output_path, "S3D_dales_オルト")
                                                       )

    success, job_id = job_manager.addSegmentation3DJob(job_name="S3D_spainPOD_オルト",
                                                       point_clouds=os.path.join(input_path,
                                                                                 r"DataSets\Spain\PointCloudsPOD"),
                                                       point_cloud_segmentation_detector=os.path.join(input_path,
                                                                                                      r"Detectors\PointCloudSegmentationDetector\RailA_v5"),
                                                       export_srs="EPSG:32629",
                                                       export_segmentation3D=[PCType.LAS],
                                                       detect_objects3D=True,
                                                       export_objects3DDGN=True,
                                                       export_objects3DCesium=True,
                                                       export_locations3DSHP=True,
                                                       output=os.path.join(output_path, "S3D_spainPOD_オルト")
                                                       )

    success, job_id = job_manager.addSegmentation3DJob(job_name="S3D_spainLAS_オルト",
                                                       point_clouds=os.path.join(input_path,
                                                                                 r"DataSets\Spain\PointCloudsLAS"),
                                                       point_cloud_segmentation_detector=os.path.join(input_path,
                                                                                                      r"Detectors\PointCloudSegmentationDetector\RailA_v5"),
                                                       export_srs="EPSG:32629",
                                                       export_segmentation3D=[PCType.LAZ, PCType.POD],
                                                       detect_objects3D=True,
                                                       export_objects3DDGN=True,
                                                       export_objects3DCesium=True,
                                                       export_locations3DSHP=True,
                                                       output=os.path.join(output_path, "S3D_spainLAS_オルト")
                                                       )

    success, job_id = job_manager.addSegmentation3DJob(job_name="S3D_spainOPC_オルト",
                                                       point_clouds=os.path.join(input_path,
                                                                                 r"DataSets\Spain\PointCloudsOPC"),
                                                       point_cloud_segmentation_detector=os.path.join(input_path,
                                                                                                      r"Detectors\PointCloudSegmentationDetector\RailA_v5"),
                                                       export_srs="EPSG:32629",
                                                       export_segmentation3D=[PCType.LAZ, PCType.POD],
                                                       detect_objects3D=True,
                                                       export_objects3DDGN=True,
                                                       export_objects3DCesium=True,
                                                       export_locations3DSHP=True,
                                                       output=os.path.join(output_path, "S3D_spainOPC_オルト")
                                                       )

    # Classifying non georeferenced point clouds and detecting objects in space
    success, job_id = job_manager.addSegmentation3DJob(job_name="S3D_spainNoGeo_オルト",
                                                       point_clouds=os.path.join(input_path,
                                                                                 r"DataSets\Spain\PointCloudsNoGeo"),
                                                       point_cloud_segmentation_detector=os.path.join(input_path,
                                                                                                      r"Detectors\PointCloudSegmentationDetector\RailA_v5"),
                                                       export_segmentation3D=[PCType.LAS],
                                                       detect_objects3D=True,
                                                       export_objects3DDGN=True,
                                                       export_objects3DCesium=True,
                                                       export_locations3DSHP=True,
                                                       output=os.path.join(output_path, "S3D_spainNoGeo_オルト")
                                                       )

    # Classifying and finding nothing to export
    success, job_id = job_manager.addSegmentation3DJob(job_name="S3D_spainNoObject_オルト",
                                                       point_clouds=os.path.join(input_path,
                                                                                 r"DataSets\Spain\PointCloudsNoGeo"),
                                                       point_cloud_segmentation_detector=os.path.join(input_path,
                                                                                                      r"Detectors\PointCloudSegmentationDetector\Test"),
                                                       export_segmentation3D=[PCType.LAS],
                                                       detect_objects3D=True,
                                                       export_objects3DDGN=True,
                                                       export_objects3DCesium=True,
                                                       export_locations3DSHP=True,
                                                       output=os.path.join(output_path, "S3D_spainNoObject_オルト")
                                                       )

    # Classifying point clouds, detecting objects in photos and using both to detect objects in space
    success, job_id = job_manager.addSegmentation3DJob(job_name="S3D_spain_refined2D_オルト",
                                                       point_clouds=os.path.join(input_path,
                                                                                 r"DataSets\Spain\PointCloudsOPC"),
                                                       point_cloud_segmentation_detector=os.path.join(input_path,
                                                                                                      r"Detectors\PointCloudSegmentationDetector\RailA_v5"),
                                                       oriented_photos=os.path.join(input_path,
                                                                                    r"DataSets\Spain\OrientedPhotos"),
                                                       photo_object_detector=os.path.join(input_path,
                                                                                          r"Detectors\PhotoObjectDetector\RailA_v5"),

                                                       export_srs="EPSG:32629",
                                                       export_segmentation3D=[PCType.LAS],
                                                       detect_objects3D=True,
                                                       export_objects3DDGN=True,
                                                       export_objects3DCesium=True,
                                                       export_locations3DSHP=True,
                                                       output=os.path.join(output_path, "S3D_spain_refined2D_オルト")
                                                       )

    # Turning meshes into classified point clouds and detecting objects in space
    success, job_id = job_manager.addSegmentation3DJob(job_name="S3D_Graz_3MX_オルト",
                                                       meshes=os.path.join(input_path,
                                                                           r"DataSets\Graz\Mesh3MX"),
                                                       point_cloud_segmentation_detector=os.path.join(input_path,
                                                                                                      r"Detectors\PointCloudSegmentationDetector\Hessigheim_v3"),
                                                       export_srs="EPSG:32633",
                                                       export_segmentation3D=[PCType.POD, PCType.LAZ],
                                                       detect_objects3D=True,
                                                       export_locations3DSHP=True,
                                                       output=os.path.join(output_path, "S3D_Graz_3MX_オルト"))

    success, job_id = job_manager.addSegmentation3DJob(job_name="S3D_Graz_3SM_オルト",
                                                       meshes=os.path.join(input_path,
                                                                           r"DataSets\Graz\Mesh3SM"),
                                                       point_cloud_segmentation_detector=os.path.join(input_path,
                                                                                                      r"Detectors\PointCloudSegmentationDetector\Hessigheim_v3"),
                                                       export_srs="EPSG:32633",
                                                       export_segmentation3D=[PCType.POD, PCType.LAZ],
                                                       detect_objects3D=True,
                                                       export_locations3DSHP=True,
                                                       output=os.path.join(output_path, "S3D_Graz_3SM_オルト")
                                                       )

    # Same as above with a ContextScene where SRS is provided instead of using 3MX or 3SM one
    success, job_id = job_manager.addSegmentation3DJob(job_name="S3D_Graz_3MX_ForcedSRS_オルト",
                                                       meshes=os.path.join(input_path,
                                                                           r"DataSets\Graz\Mesh3MX_ForcedSRS"),
                                                       point_cloud_segmentation_detector=os.path.join(input_path,
                                                                                                      r"Detectors\PointCloudSegmentationDetector\Hessigheim_v3"),
                                                       export_srs="EPSG:32633",
                                                       export_segmentation3D=[PCType.POD, PCType.LAZ],
                                                       detect_objects3D=True,
                                                       export_locations3DSHP=True,
                                                       output=os.path.join(output_path, "S3D_Graz_3MX_ForcedSRS_オルト"))

    success, job_id = job_manager.addSegmentation3DJob(job_name="S3D_Graz_3SM_ForcedSRS_オルト",
                                                       meshes=os.path.join(input_path,
                                                                           r"DataSets\Graz\Mesh3SM_ForcedSRS"),
                                                       point_cloud_segmentation_detector=os.path.join(input_path,
                                                                                                      r"Detectors\PointCloudSegmentationDetector\Hessigheim_v3"),
                                                       export_srs="EPSG:32633",
                                                       export_segmentation3D=[PCType.POD, PCType.LAZ],
                                                       detect_objects3D=True,
                                                       export_locations3DSHP=True,
                                                       output=os.path.join(output_path, "S3D_Graz_3SM_ForcedSRS_オルト")
                                                       )

    # Given classified point clouds, detecting objects in space
    success, job_id = job_manager.addSegmentation3DJob(job_name="S3D_dales_given_S3D_オルト",
                                                       segmentation3D=os.path.join(input_path,
                                                                                   r"GivenDetections\S3D_Dales"),
                                                       export_srs="EPSG:26910",
                                                       export_segmentation3D=[PCType.LAZ, PCType.POD],
                                                       detect_objects3D=True,
                                                       export_objects3DDGN=True,
                                                       export_objects3DCesium=True,
                                                       export_locations3DSHP=True,
                                                       output=os.path.join(output_path, "S3D_dales_given_S3D_オルト")
                                                       )

    success, job_id = job_manager.addSegmentation3DJob(job_name="S3D_dales_given_S3D_OPC_オルト",
                                                       segmentation3D=os.path.join(input_path,
                                                                                   r"GivenDetections\S3D_Dales_OPC\segmentation3D"),
                                                       export_srs="EPSG:26910",
                                                       export_segmentation3D=[PCType.LAZ, PCType.POD],
                                                       detect_objects3D=True,
                                                       export_objects3DDGN=True,
                                                       export_objects3DCesium=True,
                                                       export_locations3DSHP=True,
                                                       output=os.path.join(output_path, "S3D_dales_given_S3D_OPC_オルト")
                                                       )

    # Given classified point clouds and 2D objects in photos, using both to detect objects in space
    success, job_id = job_manager.addSegmentation3DJob(job_name="S3D_spain_refined2D_given_S3D_and_O2D_オルト",
                                                       segmentation3D=os.path.join(input_path,
                                                                                   r"GivenDetections\S3D_Spain"),
                                                       oriented_photos=os.path.join(input_path,
                                                                                    r"DataSets\Spain\OrientedPhotos"),
                                                       objects2D=os.path.join(input_path,
                                                                              r"GivenDetections\O2D_Spain"),
                                                       export_srs="EPSG:32629",
                                                       export_segmentation3D=[PCType.LAS],
                                                       detect_objects3D=True,
                                                       export_objects3DDGN=True,
                                                       export_objects3DCesium=True,
                                                       export_locations3DSHP=True,
                                                       output=os.path.join(output_path, "S3D_spain_refined2D_given_S3D_and_O2D_オルト")
                                                       )

    success, job_id = job_manager.addSegmentation3DJob(job_name="S3D_spain_refined2D_given_S3D_and_O2D_OPC_オルト",
                                                       segmentation3D=os.path.join(input_path,
                                                                                   r"GivenDetections\S3D_Spain_OPC\segmentation3D"),
                                                       oriented_photos=os.path.join(input_path,
                                                                                    r"DataSets\Spain\OrientedPhotos"),
                                                       objects2D=os.path.join(input_path,
                                                                              r"GivenDetections\O2D_Spain"),
                                                       export_srs="EPSG:32629",
                                                       export_segmentation3D=[PCType.LAS],
                                                       detect_objects3D=True,
                                                       export_objects3DDGN=True,
                                                       export_objects3DCesium=True,
                                                       export_locations3DSHP=True,
                                                       output=os.path.join(output_path, "S3D_spain_refined2D_given_S3D_and_O2D_OPC_オルト")
                                                       )

    # Given classified non georeferenced point clouds, detecting objects in space
    success, job_id = job_manager.addSegmentation3DJob(job_name="S3D_spainNoGeo_given_S3D_オルト",
                                                       segmentation3D=os.path.join(input_path,
                                                                                   r"GivenDetections\S3D_SpainNoGeo\segmentation3D"),
                                                       export_segmentation3D=[PCType.LAS],
                                                       detect_objects3D=True,
                                                       export_objects3DDGN=True,
                                                       export_objects3DCesium=True,
                                                       export_locations3DSHP=True,
                                                       output=os.path.join(output_path, "S3D_spainNoGeo_given_S3D_オルト")
                                                       )

    # Given a 3D segmentation confidence, export it
    success, job_id = job_manager.addSegmentation3DJob(job_name="S3D_dales_with_confidence_given_S3D_オルト",
                                                       segmentation3D=os.path.join(input_path,
                                                                                   r"GivenDetections\S3D_Dales_with_Confidence"),
                                                       export_srs="EPSG:26910",
                                                       export_segmentation3D=[PCType.LAZ, PCType.PLY],
                                                       output=os.path.join(output_path, "S3D_dales_with_confidence_given_S3D_オルト")
                                                       )

    # Classifying point clouds and detecting objects in space with TF2
    success, job_id = job_manager.addSegmentation3DJob(job_name="S3D_tuxford_オルト",
                                                       point_clouds=os.path.join(input_path,
                                                                                 r"DataSets\Tuxford\PointCloud"),
                                                       point_cloud_segmentation_detector=os.path.join(input_path,
                                                                                                      r"Detectors\PointCloudSegmentationDetector\Wingtra_v1"),
                                                       save_confidence=True,
                                                       export_segmentation3D=[PCType.LAZ, PCType.POD, PCType.PLY],
                                                       output=os.path.join(output_path, "S3D_tuxford_オルト")
                                                       )

    # Classifying point clouds and detecting objects in space with TF2 using specific tree object detection
    success, job_id = job_manager.addSegmentation3DJob(job_name="S3D_ArGodoyCruz_Tree_given_S3Dオルト",
                                                       segmentation3D=os.path.join(input_path,
                                                                                   r"GivenDetections\S3D_ArGodoyCruz\segmentation3D"),
                                                       export_srs="EPSG:5344",
                                                       detect_objects3D=True,
                                                       export_objects3DDGN=True,
                                                       export_locations3DSHP=True,
                                                       output=os.path.join(output_path,
                                                                           "S3D_ArGodoyCruz_Tree_given_S3Dオルト")
                                                       )

    success, job_id = job_manager.addSegmentation3DJob(job_name="S3D_ArGodoyCruz_Tree_extractオルト",
                                                       point_clouds=os.path.join(input_path,
                                                                                 r"DataSets\ArGodoyCruz_Tree\Extract\PointCloudsOPC"),
                                                       point_cloud_segmentation_detector=os.path.join(input_path,
                                                                                                      r"Detectors\PointCloudSegmentationDetector\Trees_v1"),
                                                       export_srs="EPSG:5344",
                                                       export_segmentation3D=[PCType.LAZ, PCType.POD],
                                                       detect_objects3D=True,
                                                       export_objects3DDGN=True,
                                                       export_locations3DSHP=True,
                                                       output=os.path.join(output_path,
                                                                           "S3D_ArGodoyCruz_Tree_extractオルト")
                                                       )

    # ========================= L3D

    # Classifying point clouds and detecting lines in space
    success, job_id = job_manager.addLines3DJob(job_name="L3D_spain_オルト",
                                                point_clouds=os.path.join(input_path,
                                                                          r"DataSets\Spain\PointCloudsOPC"),
                                                point_cloud_segmentation_detector=os.path.join(input_path,
                                                                                               r"Detectors\PointCloudSegmentationDetector\RailA_v5"),
                                                compute_line_width=True,
                                                remove_small_components=0.15,
                                                export_srs="EPSG:32629",
                                                export_lines3DDGN=True,
                                                export_lines3DCesium=True,
                                                output=os.path.join(output_path, "L3D_spain_オルト")
                                                )

    # Classifying point clouds from photo segmentation detector and detecting lines in space
    success, job_id = job_manager.addLines3DJob(job_name="L3D_BridgeWithCracks_POD_オルト",
                                                point_clouds=os.path.join(input_path,
                                                                          r"DataSets\BridgeWithCracks\PointClouds"),
                                                oriented_photos=os.path.join(input_path,
                                                                             r"DataSets\BridgeWithCracks\OrientedPhotos"),
                                                photo_segmentation_detector=os.path.join(input_path,
                                                                                         r"Detectors\PhotoSegmentationDetector\CracksA_v1"),
                                                compute_line_width=True,
                                                remove_small_components=0.15,
                                                export_lines3DDGN=True,
                                                export_lines3DCesium=True,
                                                output=os.path.join(output_path, "L3D_BridgeWithCracks_POD_オルト")
                                                )

    success, job_id = job_manager.addLines3DJob(job_name="L3D_BridgeWithCracks_OPC_オルト",
                                                point_clouds=os.path.join(input_path,
                                                                          r"DataSets\BridgeWithCracks\PointClouds_OPC"),
                                                oriented_photos=os.path.join(input_path,
                                                                             r"DataSets\BridgeWithCracks\OrientedPhotos"),
                                                photo_segmentation_detector=os.path.join(input_path,
                                                                                         r"Detectors\PhotoSegmentationDetector\CracksA_v1"),
                                                compute_line_width=True,
                                                remove_small_components=0.15,
                                                export_lines3DDGN=True,
                                                export_lines3DCesium=True,
                                                output=os.path.join(output_path, "L3D_BridgeWithCracks_OPC_オルト")
                                                )

    # Classifying 3mx mesh from photo segmentation detector and detecting lines in space
    success, job_id = job_manager.addLines3DJob(job_name="L3D_BridgeWithCracks_3MX_オルト",
                                                meshes=os.path.join(input_path,
                                                                    r"DataSets\BridgeWithCracks\Mesh3MX"),
                                                oriented_photos=os.path.join(input_path,
                                                                             r"DataSets\BridgeWithCracks\OrientedPhotos"),
                                                photo_segmentation_detector=os.path.join(input_path,
                                                                                         r"Detectors\PhotoSegmentationDetector\CracksA_v1"),
                                                compute_line_width=True,
                                                remove_small_components=0.15,
                                                export_lines3DDGN=True,
                                                export_lines3DCesium=True,
                                                output=os.path.join(output_path, "L3D_BridgeWithCracks_3MX_オルト")
                                                )

    # Classifying 3sm mesh from photo segmentation detector and detecting lines in space
    success, job_id = job_manager.addLines3DJob(job_name="L3D_BridgeWithCracks_3SM_オルト",
                                                meshes=os.path.join(input_path,
                                                                    r"DataSets\BridgeWithCracks\Mesh3SM"),
                                                oriented_photos=os.path.join(input_path,
                                                                             r"DataSets\BridgeWithCracks\OrientedPhotos"),
                                                photo_segmentation_detector=os.path.join(input_path,
                                                                                         r"Detectors\PhotoSegmentationDetector\CracksA_v1"),
                                                compute_line_width=True,
                                                remove_small_components=0.15,
                                                export_lines3DDGN=True,
                                                export_lines3DCesium=True,
                                                output=os.path.join(output_path, "L3D_BridgeWithCracks_3SM_オルト")
                                                )

    # Classifying 3mx mesh from Applied AI (Onnx format) photo segmentation detector and detecting lines in space
    success, job_id = job_manager.addLines3DJob(job_name="L3D_BridgeWithCracks_3MX_ONNX_オルト",
                                                meshes=os.path.join(input_path,
                                                                    r"DataSets\BridgeWithCracks\Mesh3MX"),
                                                oriented_photos=os.path.join(input_path,
                                                                             r"DataSets\BridgeWithCracks\OrientedPhotos"),
                                                photo_segmentation_detector=os.path.join(input_path,
                                                                                         r"Detectors\PhotoSegmentationDetector\CracksAppliedAI_v1"),
                                                compute_line_width=True,
                                                remove_small_components=0.15,
                                                export_lines3DDGN=True,
                                                export_lines3DCesium=True,
                                                output=os.path.join(output_path, "L3D_BridgeWithCracks_3MX_ONNX_オルト")
                                                )

    # Given classified point clouds, detecting lines in space
    success, job_id = job_manager.addLines3DJob(job_name="L3D_spain_given_S3D_オルト",
                                                segmentation3D=os.path.join(input_path,
                                                                            r"GivenDetections\S3D_Spain"),
                                                compute_line_width=True,
                                                remove_small_components=0.15,
                                                export_srs="EPSG:32629",
                                                export_lines3DDGN=True,
                                                export_lines3DCesium=True,
                                                output=os.path.join(output_path, "L3D_spain_given_S3D_オルト")
                                                )

    success, job_id = job_manager.addLines3DJob(job_name="L3D_spain_given_S3D_OPC_オルト",
                                                segmentation3D=os.path.join(input_path,
                                                                            r"GivenDetections\S3D_Spain_OPC\segmentation3D"),
                                                compute_line_width=True,
                                                remove_small_components=0.15,
                                                export_srs="EPSG:32629",
                                                export_lines3DDGN=True,
                                                export_lines3DCesium=True,
                                                output=os.path.join(output_path, "L3D_spain_given_S3D_OPC_オルト")
                                                )

    # Given photo segmentation, classifying point clouds and detecting lines in space
    success, job_id = job_manager.addLines3DJob(job_name="L3D_BridgeWithCracks_POD_given_S2D_オルト",
                                                point_clouds=os.path.join(input_path,
                                                                          r"DataSets\BridgeWithCracks\PointClouds"),
                                                oriented_photos=os.path.join(input_path,
                                                                             r"DataSets\BridgeWithCracks\OrientedPhotos"),
                                                segmentation2D=os.path.join(input_path,
                                                                            r"GivenDetections\S2D_BridgeWithCracks"),
                                                compute_line_width=True,
                                                remove_small_components=0.15,
                                                export_lines3DDGN=True,
                                                export_lines3DCesium=True,
                                                output=os.path.join(output_path, "L3D_BridgeWithCracks_POD_given_S2D_オルト")
                                                )

    success, job_id = job_manager.addLines3DJob(job_name="L3D_BridgeWithCracks_OPC_given_S2D_オルト",
                                                point_clouds=os.path.join(input_path,
                                                                          r"DataSets\BridgeWithCracks\PointClouds_OPC"),
                                                oriented_photos=os.path.join(input_path,
                                                                             r"DataSets\BridgeWithCracks\OrientedPhotos"),
                                                segmentation2D=os.path.join(input_path,
                                                                            r"GivenDetections\S2D_BridgeWithCracks"),
                                                compute_line_width=True,
                                                remove_small_components=0.15,
                                                export_lines3DDGN=True,
                                                export_lines3DCesium=True,
                                                output=os.path.join(output_path, "L3D_BridgeWithCracks_OPC_given_S2D_オルト")
                                                )

    # Given photo segmentation, classifying 3MX mesh and detecting lines in space
    success, job_id = job_manager.addLines3DJob(job_name="L3D_BridgeWithCracks_3MX_given_S2D_オルト",
                                                meshes=os.path.join(input_path,
                                                                    r"DataSets\BridgeWithCracks\Mesh3MX"),
                                                oriented_photos=os.path.join(input_path,
                                                                             r"DataSets\BridgeWithCracks\OrientedPhotos"),
                                                segmentation2D=os.path.join(input_path,
                                                                            r"GivenDetections\S2D_BridgeWithCracks"),
                                                compute_line_width=True,
                                                remove_small_components=0.15,
                                                export_lines3DDGN=True,
                                                export_lines3DCesium=True,
                                                output=os.path.join(output_path, "L3D_BridgeWithCracks_3MX_given_S2D_オルト")
                                                )

    # Given photo segmentation, classifying 3SM mesh and detecting lines in space
    success, job_id = job_manager.addLines3DJob(job_name="L3D_BridgeWithCracks_3SM_given_S2D_オルト",
                                                meshes=os.path.join(input_path,
                                                                    r"DataSets\BridgeWithCracks\Mesh3SM"),
                                                oriented_photos=os.path.join(input_path,
                                                                             r"DataSets\BridgeWithCracks\OrientedPhotos"),
                                                segmentation2D=os.path.join(input_path,
                                                                            r"GivenDetections\S2D_BridgeWithCracks"),
                                                compute_line_width=True,
                                                remove_small_components=0.15,
                                                export_lines3DDGN=True,
                                                export_lines3DCesium=True,
                                                output=os.path.join(output_path, "L3D_BridgeWithCracks_3SM_given_S2D_オルト")
                                                )

    # ========================= Change Detection

    # Detecting changes between two point cloud collections
    success, job_id = job_manager.addChangeDetectionJob(job_name="Change_spain_オルト",
                                                        point_clouds1=os.path.join(input_path,
                                                                                   r"DataSets\SpainChange\PointClouds1"),
                                                        point_clouds2=os.path.join(input_path,
                                                                                   r"DataSets\SpainChange\PointClouds2"),
                                                        save_point_clouds=True,
                                                        color_threshold_low=0.3,
                                                        color_threshold_high=0.6,
                                                        min_points=50,
                                                        output=os.path.join(output_path, "Change_spain_オルト")
                                                        )

    success, job_id = job_manager.addChangeDetectionJob(job_name="Change_horn_local_オルト",
                                                        point_clouds1=os.path.join(input_path,
                                                                                   r"DataSets\Horn\NoSRS\PointClouds1"),
                                                        point_clouds2=os.path.join(input_path,
                                                                                   r"DataSets\Horn\NoSRS\PointClouds2"),
                                                        save_point_clouds=True,
                                                        dist_threshold_low=5,
                                                        dist_threshold_high=15,
                                                        min_points=1000,
                                                        output=os.path.join(output_path, "Change_horn_local_オルト")
                                                        )

    success, job_id = job_manager.addChangeDetectionJob(job_name="Change_horn_las_オルト",
                                                        point_clouds1=os.path.join(input_path,
                                                                                   r"DataSets\Horn\LAZ\PointClouds1"),
                                                        point_clouds2=os.path.join(input_path,
                                                                                   r"DataSets\Horn\LAZ\PointClouds2"),
                                                        save_point_clouds=True,
                                                        dist_threshold_low=5,
                                                        dist_threshold_high=15,
                                                        min_points=1000,
                                                        export_srs="EPSG:32633",
                                                        export_locations3DSHP=True,
                                                        output=os.path.join(output_path, "Change_horn_las_オルト")
                                                        )

    success, job_id = job_manager.addChangeDetectionJob(job_name="Change_horn_pod_オルト",
                                                        point_clouds1=os.path.join(input_path,
                                                                                   r"DataSets\Horn\POD\PointClouds1"),
                                                        point_clouds2=os.path.join(input_path,
                                                                                   r"DataSets\Horn\POD\PointClouds2"),
                                                        save_point_clouds=True,
                                                        dist_threshold_low=5,
                                                        dist_threshold_high=15,
                                                        min_points=1000,
                                                        export_srs="EPSG:32633",
                                                        export_locations3DSHP=True,
                                                        output=os.path.join(output_path, "Change_horn_pod_オルト")
                                                        )

    success, job_id = job_manager.addChangeDetectionJob(job_name="Change_horn_ply_オルト",
                                                        point_clouds1=os.path.join(input_path,
                                                                                   r"DataSets\Horn\PLY\PointClouds1"),
                                                        point_clouds2=os.path.join(input_path,
                                                                                   r"DataSets\Horn\PLY\PointClouds2"),
                                                        save_point_clouds=True,
                                                        dist_threshold_low=5,
                                                        dist_threshold_high=15,
                                                        min_points=1000,
                                                        output=os.path.join(output_path, "Change_horn_ply_オルト")
                                                        )

    success, job_id = job_manager.addChangeDetectionJob(job_name="Change_horn_opc_オルト",
                                                        point_clouds1=os.path.join(input_path,
                                                                                   r"DataSets\Horn\OPC\PointClouds1"),
                                                        point_clouds2=os.path.join(input_path,
                                                                                   r"DataSets\Horn\OPC\PointClouds2"),
                                                        save_point_clouds=True,
                                                        dist_threshold_low=5,
                                                        dist_threshold_high=15,
                                                        min_points=1000,
                                                        output=os.path.join(output_path, "Change_horn_opc_オルト")
                                                        )

    # Detecting changes between two mesh collections
    success, job_id = job_manager.addChangeDetectionJob(job_name="Change_lake_オルト",
                                                        meshes1=os.path.join(input_path,
                                                                             r"DataSets\Lake\Meshes1"),
                                                        meshes2=os.path.join(input_path,
                                                                             r"DataSets\Lake\Meshes2"),
                                                        save_point_clouds=True,
                                                        dist_threshold_low=5,
                                                        dist_threshold_high=15,
                                                        min_points=1000,
                                                        resolution=0.5,
                                                        output=os.path.join(output_path, "Change_lake_オルト")
                                                        )

# ============================== Submit cloud jobs ====================

if not nocloud:

    # ========================= O2D

    # Detecting objects in photos on the cloud
    success, job_id = job_manager.addObjects2DJob(job_name="O2D_motos_cloud",
                                                  process_on_cloud=True,
                                                  itwin_id=itwin_id,
                                                  photos=os.path.join(input_path,
                                                                      r"DataSets\Motos\Photos"),
                                                  photo_object_detector=os.path.join(input_path,
                                                                                 r"Detectors\PhotoObjectDetector\Coco2017_v1"),
                                                  output=os.path.join(output_path, "O2D_motos_cloud")
                                                  )

    success, job_id = job_manager.addObjects2DJob(job_name="O2D_moto_low_confidence_cloud",
                                                  process_on_cloud=True,
                                                  itwin_id=itwin_id,
                                                  photos=os.path.join(input_path,
                                                                      r"DataSets\Motos\Photos"),
                                                  photo_object_detector=os.path.join(input_path,
                                                                                     r"Detectors\PhotoObjectDetector\Coco2017_low_confidence_v1"),
                                                  output=os.path.join(output_path, "O2D_moto_low_confidence_cloud")
                                                  )

    # ========================= S2D

    # Classifying pixels in photos on the cloud
    success, job_id = job_manager.addSegmentation2DJob(job_name="S2D_motos_cloud",
                                                       process_on_cloud=True,
                                                       itwin_id=itwin_id,
                                                       photos=os.path.join(input_path,
                                                                           r"DataSets\Motos\Photos"),
                                                       photo_segmentation_detector=os.path.join(input_path,
                                                                                                r"Detectors\PhotoSegmentationDetector\PascalVoc2012_v1"),
                                                       output=os.path.join(output_path, "S2D_motos_cloud")
                                                       )

    success, job_id = job_manager.addSegmentation2DJob(job_name="S2D_BridgeWithCracks_cloud",
                                                       process_on_cloud=True,
                                                       itwin_id=itwin_id,
                                                       photos=os.path.join(input_path,
                                                                                   r"DataSets\BridgeWithCracks\OrientedPhotos"),
                                                       photo_segmentation_detector=os.path.join(input_path,
                                                                                               r"Detectors\PhotoSegmentationDetector\CracksA_v1"),
                                                       output=os.path.join(output_path, "S2D_BridgeWithCracks_cloud")
                                                       )

    success, job_id = job_manager.addSegmentation2DJob(job_name="S2D_BridgeWithCracks_ONNX_cloud",
                                                       process_on_cloud=True,
                                                       itwin_id=itwin_id,
                                                       photos=os.path.join(input_path,
                                                                           r"DataSets\BridgeWithCracks\OrientedPhotos"),
                                                       photo_segmentation_detector=os.path.join(input_path,
                                                                                                r"Detectors\PhotoSegmentationDetector\CracksAppliedAI_v1"),
                                                       output=os.path.join(output_path,
                                                                           "S2D_BridgeWithCracks_ONNX_cloud")
                                                       )

    # Classifying pixels in orthophoto on the cloud and exporting 2D polygons
    success, job_id = job_manager.addSegmentation2DJob(job_name="S2D_christchurch_cloud",
                                                       process_on_cloud=True,
                                                       itwin_id=itwin_id,
                                                       orthophoto=os.path.join(input_path,
                                                                               r"DataSets\Christchurch\Orthophoto"),
                                                       orthophoto_segmentation_detector=os.path.join(input_path,
                                                                                                 r"Detectors\OrthophotoSegmentationDetector\RGB\RoofB_v3"),
                                                       detect_polygons2D=True,
                                                       export_polygons2D=True,
                                                       output=os.path.join(output_path, "S2D_christchurch_cloud")
                                                       )

    success, job_id = job_manager.addSegmentation2DJob(job_name="S2D_christchurch_JPEG_cloud",
                                                       process_on_cloud=True,
                                                       itwin_id=itwin_id,
                                                       orthophoto=os.path.join(input_path,
                                                                               r"DataSets\Christchurch_JPEG\Orthophoto"),
                                                       orthophoto_segmentation_detector=os.path.join(input_path,
                                                                                                     r"Detectors\OrthophotoSegmentationDetector\RGB\RoofB_v3"),
                                                       detect_polygons2D=True,
                                                       export_polygons2D=True,
                                                       output=os.path.join(output_path, "S2D_christchurch_JPEG_cloud")
                                                       )

    # Classifing pixels in orthophoto on the cloud dand detecting 2D lines
    success, job_id = job_manager.addSegmentation2DJob(job_name="S2D_MinnesotaDOT_cloud",
                                                       process_on_cloud=True,
                                                       itwin_id=itwin_id,
                                                       orthophoto=os.path.join(input_path,
                                                                               r"DataSets\MinnesotaDOT\OrthophotosScene"),
                                                       orthophoto_segmentation_detector=os.path.join(input_path,
                                                                                                     r"Detectors\OrthophotoSegmentationDetector\RGB\CracksA_v1"),
                                                       detect_lines2D=True,
                                                       output=os.path.join(output_path, "S2D_MinnesotaDOT_cloud")
                                                       )

    success, job_id = job_manager.addSegmentation2DJob(job_name="S2D_MinnesotaDOT_Onnx_cloud",
                                                       process_on_cloud=True,
                                                       itwin_id=itwin_id,
                                                       orthophoto=os.path.join(input_path,
                                                                               r"DataSets\MinnesotaDOT\OrthophotosScene"),
                                                       orthophoto_segmentation_detector=os.path.join(input_path,
                                                                                                     r"Detectors\OrthophotoSegmentationDetector\RGB\CracksAppliedAI_v1"),
                                                       detect_lines2D=True,
                                                       output=os.path.join(output_path, "S2D_MinnesotaDOT_Onnx_cloud")
                                                       )

    # Classifying pixels in orthophoto with DSM on the cloud and exporting 2D polygons
    success, job_id = job_manager.addSegmentation2DJob(job_name="S2D_Graz_cloud",
                                                       process_on_cloud=True,
                                                       itwin_id=itwin_id,
                                                       orthophoto=os.path.join(input_path,
                                                                               r"DataSets\Graz\OrthophotoScene"),
                                                       orthophoto_segmentation_detector=os.path.join(input_path,
                                                                                                    r"Detectors\OrthophotoSegmentationDetector\RGB_DSM\CityA_RGBD_v1"),
                                                       detect_polygons2D=True,
                                                       export_polygons2D=True,
                                                       output=os.path.join(output_path, "S2D_Graz_cloud"),
                                                       )

    # ========================= O3D

    # Detecting objects in photos and objects in point clouds
    success, job_id = job_manager.addObjects3DJob(job_name="O3D_spain_cloud",
                                                  process_on_cloud=True,
                                                  itwin_id=itwin_id,
                                                  oriented_photos=os.path.join(input_path,
                                                                               r"DataSets\Spain\OrientedPhotos"),
                                                  point_clouds=os.path.join(input_path,
                                                                            r"DataSets\Spain\PointCloudsPOD"),
                                                  photo_object_detector=os.path.join(input_path,
                                                                                     r"Detectors\PhotoObjectDetector\RailA_v5"),
                                                  export_srs="EPSG:32629",
                                                  export_objects3DDGN=True,
                                                  export_objects3DCesium=True,
                                                  export_locations3DSHP=True,
                                                  output=os.path.join(output_path, "O3D_spain_cloud")
                                                  )

    success, job_id = job_manager.addObjects3DJob(job_name="O3D_spain_OPC_cloud",
                                                  process_on_cloud=True,
                                                  itwin_id=itwin_id,
                                                  oriented_photos=os.path.join(input_path,
                                                                               r"DataSets\Spain\OrientedPhotos"),
                                                  point_clouds=os.path.join(input_path,
                                                                            r"DataSets\Spain\PointCloudsOPC"),
                                                  photo_object_detector=os.path.join(input_path,
                                                                                     r"Detectors\PhotoObjectDetector\RailA_v5"),
                                                  export_srs="EPSG:32629",
                                                  export_objects3DDGN=True,
                                                  export_objects3DCesium=True,
                                                  export_locations3DSHP=True,
                                                  output=os.path.join(output_path, "O3D_spain_OPC_cloud")
                                                  )

    # Detecting objects in photos and objects in space
    success, job_id = job_manager.addObjects3DJob(job_name="O3D_antennas_cloud",
                                                  process_on_cloud=True,
                                                  itwin_id=itwin_id,
                                                  oriented_photos=os.path.join(input_path,
                                                                               r"DataSets\Antennas\OrientedPhotos"),
                                                  photo_object_detector=os.path.join(input_path,
                                                                                     r"Detectors\PhotoObjectDetector\Antennas_v1"),
                                                  export_objects3DDGN=True,
                                                  export_objects3DCesium=True,
                                                  export_locations3DSHP=True,
                                                  output=os.path.join(output_path, "O3D_antennas_cloud")
                                                  )

    # Given objects in photos, detecting objects in point clouds
    success, job_id = job_manager.addObjects3DJob(job_name="O3D_spain_given_O2D_cloud",
                                                  process_on_cloud=True,
                                                  itwin_id=itwin_id,
                                                  oriented_photos=os.path.join(input_path,
                                                                               r"DataSets\Spain\OrientedPhotos"),
                                                  point_clouds=os.path.join(input_path,
                                                                            r"DataSets\Spain\PointCloudsPOD"),
                                                  objects2D=os.path.join(input_path,
                                                                         r"GivenDetections\O2D_Spain"),
                                                  export_srs="EPSG:32629",
                                                  export_objects3DDGN=True,
                                                  export_objects3DCesium=True,
                                                  export_locations3DSHP=True,
                                                  output=os.path.join(output_path, "O3D_spain_given_O2D_cloud")
                                                  )

    success, job_id = job_manager.addObjects3DJob(job_name="O3D_spain_given_O2D_OPC_cloud",
                                                  process_on_cloud=True,
                                                  itwin_id=itwin_id,
                                                  oriented_photos=os.path.join(input_path,
                                                                               r"DataSets\Spain\OrientedPhotos"),
                                                  point_clouds=os.path.join(input_path,
                                                                            r"DataSets\Spain\PointCloudsOPC"),
                                                  objects2D=os.path.join(input_path,
                                                                         r"GivenDetections\O2D_Spain"),
                                                  export_srs="EPSG:32629",
                                                  export_objects3DDGN=True,
                                                  export_objects3DCesium=True,
                                                  export_locations3DSHP=True,
                                                  output=os.path.join(output_path, "O3D_spain_given_O2D_OPC_cloud")
                                                  )

    # Given objects in photos, detecting objects in space
    success, job_id = job_manager.addObjects3DJob(job_name="O3D_antennas_given_O2D_cloud",
                                                  process_on_cloud=True,
                                                  itwin_id=itwin_id,
                                                  oriented_photos=os.path.join(input_path,
                                                                               r"DataSets\Antennas\OrientedPhotos"),
                                                  objects2D=os.path.join(input_path,
                                                                         r"GivenDetections\O2D_Antennas"),
                                                  export_objects3DDGN=True,
                                                  export_objects3DCesium=True,
                                                  export_locations3DSHP=True,
                                                  output=os.path.join(output_path, "O3D_antennas_given_O2D_cloud")
                                                  )

    # ========================= S3D

    # Classifying point clouds and detecting objects in space
    success, job_id = job_manager.addSegmentation3DJob(job_name="S3D_dales_cloud",
                                                       process_on_cloud=True,
                                                       itwin_id=itwin_id,
                                                       point_clouds=os.path.join(input_path,
                                                                                 r"DataSets\Dales\Pointclouds"),
                                                       point_cloud_segmentation_detector=os.path.join(input_path,
                                                                                                      r"Detectors\PointCloudSegmentationDetector\Dales_v2"),
                                                       save_confidence=True,
                                                       export_srs="EPSG:26910",
                                                       export_segmentation3D=[PCType.LAZ, PCType.POD, PCType.PLY],
                                                       detect_objects3D=True,
                                                       export_objects3DDGN=True,
                                                       export_objects3DCesium=True,
                                                       export_objects3D_to_OBJ=True,
                                                       export_locations3DSHP=True,
                                                       output=os.path.join(output_path, "S3D_dales_cloud")
                                                       )

    success, job_id = job_manager.addSegmentation3DJob(job_name="S3D_spainPOD_cloud",
                                                       process_on_cloud=True,
                                                       itwin_id=itwin_id,
                                                       point_clouds=os.path.join(input_path,
                                                                                 r"DataSets\Spain\PointCloudsPOD"),
                                                       point_cloud_segmentation_detector=os.path.join(input_path,
                                                                                                      r"Detectors\PointCloudSegmentationDetector\RailA_v5"),
                                                       export_srs="EPSG:32629",
                                                       export_segmentation3D=[PCType.LAS],
                                                       detect_objects3D=True,
                                                       export_objects3DDGN=True,
                                                       export_objects3DCesium=True,
                                                       export_objects3D_to_point_clouds=PCType.PLY,
                                                       export_locations3DSHP=True,
                                                       output=os.path.join(output_path, "S3D_spainPOD_cloud")
                                                       )

    success, job_id = job_manager.addSegmentation3DJob(job_name="S3D_spainLAS_cloud",
                                                       process_on_cloud=True,
                                                       itwin_id=itwin_id,
                                                       point_clouds=os.path.join(input_path,
                                                                                 r"DataSets\Spain\PointCloudsLAS"),
                                                       point_cloud_segmentation_detector=os.path.join(input_path,
                                                                                                      r"Detectors\PointCloudSegmentationDetector\RailA_v5"),
                                                       export_srs="EPSG:32629",
                                                       export_segmentation3D=[PCType.LAZ, PCType.POD],
                                                       detect_objects3D=True,
                                                       export_objects3DDGN=True,
                                                       export_objects3DCesium=True,
                                                       export_locations3DSHP=True,
                                                       output=os.path.join(output_path, "S3D_spainLAS_cloud")
                                                       )

    success, job_id = job_manager.addSegmentation3DJob(job_name="S3D_spainOPC_cloud",
                                                       process_on_cloud=True,
                                                       itwin_id=itwin_id,
                                                       point_clouds=os.path.join(input_path,
                                                                                 r"DataSets\Spain\PointCloudsOPC"),
                                                       point_cloud_segmentation_detector=os.path.join(input_path,
                                                                                                      r"Detectors\PointCloudSegmentationDetector\RailA_v5"),
                                                       export_srs="EPSG:32629",
                                                       export_segmentation3D=[PCType.LAZ, PCType.POD],
                                                       detect_objects3D=True,
                                                       export_objects3DDGN=True,
                                                       export_objects3DCesium=True,
                                                       export_locations3DSHP=True,
                                                       output=os.path.join(output_path, "S3D_spainOPC_cloud")
                                                       )

    # Classifying point clouds, detecting objects in photos and using both to detect objects in space
    success, job_id = job_manager.addSegmentation3DJob(job_name="S3D_spain_refined2D_cloud",
                                                       process_on_cloud=True,
                                                       itwin_id=itwin_id,
                                                       point_clouds=os.path.join(input_path,
                                                                                 r"DataSets\Spain\PointCloudsOPC"),
                                                       point_cloud_segmentation_detector=os.path.join(input_path,
                                                                                                      r"Detectors\PointCloudSegmentationDetector\RailA_v5"),
                                                       oriented_photos=os.path.join(input_path,
                                                                                    r"DataSets\Spain\OrientedPhotos"),
                                                       photo_object_detector=os.path.join(input_path,
                                                                                          r"Detectors\PhotoObjectDetector\RailA_v5"),

                                                       export_srs="EPSG:32629",
                                                       export_segmentation3D=[PCType.LAS],
                                                       detect_objects3D=True,
                                                       export_objects3DDGN=True,
                                                       export_objects3DCesium=True,
                                                       export_locations3DSHP=True,
                                                       output=os.path.join(output_path, "S3D_spain_refined2D_cloud")
                                                       )

    # Classifying non georeferenced point clouds and detecting objects in space
    success, job_id = job_manager.addSegmentation3DJob(job_name="S3D_spainNoGeo_cloud",
                                                       process_on_cloud=True,
                                                       itwin_id=itwin_id,
                                                       point_clouds=os.path.join(input_path,
                                                                                 r"DataSets\Spain\PointCloudsNoGeo"),
                                                       point_cloud_segmentation_detector=os.path.join(input_path,
                                                                                                      r"Detectors\PointCloudSegmentationDetector\RailA_v5"),
                                                       export_segmentation3D=[PCType.LAS],
                                                       detect_objects3D=True,
                                                       export_objects3DDGN=True,
                                                       export_objects3DCesium=True,
                                                       export_locations3DSHP=True,
                                                       output=os.path.join(output_path, "S3D_spainNoGeo_cloud")
                                                       )

    # Classifying and finding nothing to export
    success, job_id = job_manager.addSegmentation3DJob(job_name="S3D_spainNoObject_cloud",
                                                       process_on_cloud=True,
                                                       itwin_id=itwin_id,
                                                       point_clouds=os.path.join(input_path,
                                                                                 r"DataSets\Spain\PointCloudsNoGeo"),
                                                       point_cloud_segmentation_detector=os.path.join(input_path,
                                                                                                      r"Detectors\PointCloudSegmentationDetector\Test"),
                                                       export_segmentation3D=[PCType.LAS],
                                                       detect_objects3D=True,
                                                       export_objects3DDGN=True,
                                                       export_objects3DCesium=True,
                                                       export_locations3DSHP=True,
                                                       output=os.path.join(output_path, "S3D_spainNoObject_cloud")
                                                       )

    # Turning meshes into classified point clouds and detecting objects in space
    success, job_id = job_manager.addSegmentation3DJob(job_name="S3D_Graz_3MX_cloud",
                                                       process_on_cloud=True,
                                                       itwin_id=itwin_id,
                                                       meshes=os.path.join(input_path,
                                                                           r"DataSets\Graz\Mesh3MX"),
                                                       point_cloud_segmentation_detector=os.path.join(input_path,
                                                                                                      r"Detectors\PointCloudSegmentationDetector\Hessigheim_v3"),
                                                       export_srs="EPSG:32633",
                                                       export_segmentation3D=[PCType.POD, PCType.LAZ],
                                                       detect_objects3D=True,
                                                       export_locations3DSHP=True,
                                                       output=os.path.join(output_path, "S3D_Graz_3MX_cloud"))

    success, job_id = job_manager.addSegmentation3DJob(job_name="S3D_Graz_3SM_cloud",
                                                       process_on_cloud=True,
                                                       itwin_id=itwin_id,
                                                       meshes=os.path.join(input_path,
                                                                           r"DataSets\Graz\Mesh3SM"),
                                                       point_cloud_segmentation_detector=os.path.join(input_path,
                                                                                                      r"Detectors\PointCloudSegmentationDetector\Hessigheim_v3"),
                                                       export_srs="EPSG:32633",
                                                       export_segmentation3D=[PCType.POD, PCType.LAZ],
                                                       detect_objects3D=True,
                                                       export_locations3DSHP=True,
                                                       output=os.path.join(output_path, "S3D_Graz_3SM_cloud")
                                                       )

    # Same as above with a ContextScene where SRS is provided instead of using 3MX or 3SM one
    success, job_id = job_manager.addSegmentation3DJob(job_name="S3D_Graz_3MX_ForcedSRS_cloud",
                                                       process_on_cloud=True,
                                                       itwin_id=itwin_id,
                                                       meshes=os.path.join(input_path,
                                                                           r"DataSets\Graz\Mesh3MX_ForcedSRS"),
                                                       point_cloud_segmentation_detector=os.path.join(input_path,
                                                                                                      r"Detectors\PointCloudSegmentationDetector\Hessigheim_v3"),
                                                       export_srs="EPSG:32633",
                                                       export_segmentation3D=[PCType.POD, PCType.LAZ],
                                                       detect_objects3D=True,
                                                       export_locations3DSHP=True,
                                                       output=os.path.join(output_path, "S3D_Graz_3MX_ForcedSRS_cloud"))

    success, job_id = job_manager.addSegmentation3DJob(job_name="S3D_Graz_3SM_ForcedSRS_cloud",
                                                       process_on_cloud=True,
                                                       itwin_id=itwin_id,
                                                       meshes=os.path.join(input_path,
                                                                           r"DataSets\Graz\Mesh3SM_ForcedSRS"),
                                                       point_cloud_segmentation_detector=os.path.join(input_path,
                                                                                                      r"Detectors\PointCloudSegmentationDetector\Hessigheim_v3"),
                                                       export_srs="EPSG:32633",
                                                       export_segmentation3D=[PCType.POD, PCType.LAZ],
                                                       detect_objects3D=True,
                                                       export_locations3DSHP=True,
                                                       output=os.path.join(output_path, "S3D_Graz_3SM_ForcedSRS_cloud")
                                                       )

    # Given classified point clouds, detecting objects in space
    success, job_id = job_manager.addSegmentation3DJob(job_name="S3D_dales_given_S3D_cloud",
                                                       process_on_cloud=True,
                                                       itwin_id=itwin_id,
                                                       segmentation3D=os.path.join(input_path,
                                                                                   r"GivenDetections\S3D_Dales"),
                                                       export_srs="EPSG:26910",
                                                       export_segmentation3D=[PCType.LAZ, PCType.POD],
                                                       detect_objects3D=True,
                                                       export_objects3DDGN=True,
                                                       export_objects3DCesium=True,
                                                       export_locations3DSHP=True,
                                                       output=os.path.join(output_path, "S3D_dales_given_S3D_cloud")
                                                       )

    success, job_id = job_manager.addSegmentation3DJob(job_name="S3D_dales_given_S3D_OPC_cloud",
                                                       process_on_cloud=True,
                                                       itwin_id=itwin_id,
                                                       segmentation3D=os.path.join(input_path,
                                                                                   r"GivenDetections\S3D_Dales_OPC\segmentation3D"),
                                                       export_srs="EPSG:26910",
                                                       export_segmentation3D=[PCType.LAZ, PCType.POD],
                                                       detect_objects3D=True,
                                                       export_objects3DDGN=True,
                                                       export_objects3DCesium=True,
                                                       export_locations3DSHP=True,
                                                       output=os.path.join(output_path, "S3D_dales_given_S3D_OPC_cloud")
                                                       )

    # Given classified non georeferenced point clouds, detecting objects in space
    success, job_id = job_manager.addSegmentation3DJob(job_name="S3D_spainNoGeo_given_S3D_cloud",
                                                       process_on_cloud=True,
                                                       itwin_id=itwin_id,
                                                       segmentation3D=os.path.join(input_path,
                                                                                   r"GivenDetections\S3D_SpainNoGeo\segmentation3D"),
                                                       export_segmentation3D=[PCType.LAS],
                                                       detect_objects3D=True,
                                                       export_objects3DDGN=True,
                                                       export_objects3DCesium=True,
                                                       export_locations3DSHP=True,
                                                       output=os.path.join(output_path, "S3D_spainNoGeo_given_S3D_cloud")
                                                       )

    # Given classified point clouds and 2D objects in photos, using both to detect objects in space
    success, job_id = job_manager.addSegmentation3DJob(job_name="S3D_spain_refined2D_given_S3D_and_O2D_cloud",
                                                       process_on_cloud=True,
                                                       itwin_id=itwin_id,
                                                       segmentation3D=os.path.join(input_path,
                                                                                   r"GivenDetections\S3D_Spain"),
                                                       oriented_photos=os.path.join(input_path,
                                                                                    r"DataSets\Spain\OrientedPhotos"),
                                                       objects2D=os.path.join(input_path,
                                                                              r"GivenDetections\O2D_Spain"),
                                                       export_srs="EPSG:32629",
                                                       export_segmentation3D=[PCType.LAS],
                                                       detect_objects3D=True,
                                                       export_objects3DDGN=True,
                                                       export_objects3DCesium=True,
                                                       export_locations3DSHP=True,
                                                       output=os.path.join(output_path, "S3D_spain_refined2D_given_S3D_and_O2D_cloud")
                                                       )

    success, job_id = job_manager.addSegmentation3DJob(job_name="S3D_spain_refined2D_given_S3D_and_O2D_OPC_cloud",
                                                       process_on_cloud=True,
                                                       itwin_id=itwin_id,
                                                       segmentation3D=os.path.join(input_path,
                                                                                   r"GivenDetections\S3D_Spain_OPC\segmentation3D"),
                                                       oriented_photos=os.path.join(input_path,
                                                                                    r"DataSets\Spain\OrientedPhotos"),
                                                       objects2D=os.path.join(input_path,
                                                                              r"GivenDetections\O2D_Spain"),
                                                       export_srs="EPSG:32629",
                                                       export_segmentation3D=[PCType.LAS],
                                                       detect_objects3D=True,
                                                       export_objects3DDGN=True,
                                                       export_objects3DCesium=True,
                                                       export_locations3DSHP=True,
                                                       output=os.path.join(output_path, "S3D_spain_refined2D_given_S3D_and_O2D_OPC_cloud")
                                                       )

    # Given a 3D segmentation confidence, export it
    success, job_id = job_manager.addSegmentation3DJob(job_name="S3D_dales_with_confidence_given_S3D_cloud",
                                                       process_on_cloud=True,
                                                       itwin_id=itwin_id,
                                                       segmentation3D=os.path.join(input_path,
                                                                                   r"GivenDetections\S3D_Dales_with_Confidence"),
                                                       export_srs="EPSG:26910",
                                                       export_segmentation3D=[PCType.LAZ, PCType.PLY],
                                                       output=os.path.join(output_path, "S3D_dales_with_confidence_given_S3D_cloud")
                                                       )

    # Classifying point clouds and detecting objects in space with TF2
    success, job_id = job_manager.addSegmentation3DJob(job_name="S3D_tuxford_cloud",
                                                       process_on_cloud=True,
                                                       itwin_id=itwin_id,
                                                       point_clouds=os.path.join(input_path,
                                                                                 r"DataSets\Tuxford\PointCloud"),
                                                       point_cloud_segmentation_detector=os.path.join(input_path,
                                                                                                      r"Detectors\PointCloudSegmentationDetector\Wingtra_v1"),
                                                       save_confidence=True,
                                                       export_segmentation3D=[PCType.LAZ, PCType.POD, PCType.PLY],
                                                       output=os.path.join(output_path, "S3D_tuxford_cloud")
                                                       )

    # Classifying point clouds and detecting objects in space with TF2 using specific tree object detection
    success, job_id = job_manager.addSegmentation3DJob(job_name="S3D_ArGodoyCruz_Tree_given_S3D_cloud",
                                                       process_on_cloud=True,
                                                       itwin_id=itwin_id,
                                                       segmentation3D=os.path.join(input_path,
                                                                                   r"GivenDetections\S3D_ArGodoyCruz\segmentation3D"),
                                                       export_srs="EPSG:5344",
                                                       detect_objects3D=True,
                                                       export_objects3DDGN=True,
                                                       export_locations3DSHP=True,
                                                       output=os.path.join(output_path,
                                                                           "S3D_ArGodoyCruz_Tree_given_S3D_cloud")
                                                       )

    success, job_id = job_manager.addSegmentation3DJob(job_name="S3D_ArGodoyCruz_Tree_extract_cloud",
                                                       process_on_cloud=True,
                                                       itwin_id=itwin_id,
                                                       point_clouds=os.path.join(input_path,
                                                                                 r"DataSets\ArGodoyCruz_Tree\Extract\PointCloudsOPC"),
                                                       point_cloud_segmentation_detector=os.path.join(input_path,
                                                                                                      r"Detectors\PointCloudSegmentationDetector\Trees_v1"),
                                                       export_srs="EPSG:5344",
                                                       export_segmentation3D=[PCType.LAZ, PCType.POD],
                                                       detect_objects3D=True,
                                                       export_objects3DDGN=True,
                                                       export_locations3DSHP=True,
                                                       output=os.path.join(output_path,
                                                                           "S3D_ArGodoyCruz_Tree_extract_cloud")
                                                       )

    # ========================= L3D

    # Classifying point clouds and detecting lines in space
    success, job_id = job_manager.addLines3DJob(job_name="L3D_spain_cloud",
                                                process_on_cloud=True,
                                                itwin_id=itwin_id,
                                                point_clouds=os.path.join(input_path,
                                                                          r"DataSets\Spain\PointCloudsOPC"),
                                                point_cloud_segmentation_detector=os.path.join(input_path,
                                                                                               r"Detectors\PointCloudSegmentationDetector\RailA_v5"),
                                                compute_line_width=True,
                                                remove_small_components=0.15,
                                                export_srs="EPSG:32629",
                                                export_lines3DDGN=True,
                                                export_lines3DCesium=True,
                                                output=os.path.join(output_path, "L3D_spain_cloud")
                                                )

    # Classifying point clouds from photo segmentation detector and detecting lines in space
    success, job_id = job_manager.addLines3DJob(job_name="L3D_BridgeWithCracks_POD_cloud",
                                                process_on_cloud=True,
                                                itwin_id=itwin_id,
                                                point_clouds=os.path.join(input_path,
                                                                          r"DataSets\BridgeWithCracks\PointClouds"),
                                                oriented_photos=os.path.join(input_path,
                                                                             r"DataSets\BridgeWithCracks\OrientedPhotos"),
                                                photo_segmentation_detector=os.path.join(input_path,
                                                                                         r"Detectors\PhotoSegmentationDetector\CracksA_v1"),
                                                compute_line_width=True,
                                                remove_small_components=0.15,
                                                export_lines3DDGN=True,
                                                export_lines3DCesium=True,
                                                output=os.path.join(output_path, "L3D_BridgeWithCracks_POD_cloud")
                                                )

    success, job_id = job_manager.addLines3DJob(job_name="L3D_BridgeWithCracks_OPC_cloud",
                                                process_on_cloud=True,
                                                itwin_id=itwin_id,
                                                point_clouds=os.path.join(input_path,
                                                                          r"DataSets\BridgeWithCracks\PointClouds_OPC"),
                                                oriented_photos=os.path.join(input_path,
                                                                             r"DataSets\BridgeWithCracks\OrientedPhotos"),
                                                photo_segmentation_detector=os.path.join(input_path,
                                                                                         r"Detectors\PhotoSegmentationDetector\CracksA_v1"),
                                                compute_line_width=True,
                                                remove_small_components=0.15,
                                                export_lines3DDGN=True,
                                                export_lines3DCesium=True,
                                                output=os.path.join(output_path, "L3D_BridgeWithCracks_OPC_cloud")
                                                )
    # Same but with a temporary CCOrientation workaround
    success, job_id = job_manager.addLines3DJob(job_name="L3D_BridgeWithCracks_CCOrientations_cloud",
                                                process_on_cloud=True,
                                                itwin_id=itwin_id,
                                                point_clouds=os.path.join(input_path,
                                                                          r"DataSets\BridgeWithCracks\PointClouds_OPC"),
                                                oriented_photos=os.path.join(input_path,
                                                                             r"DataSets\BridgeWithCracks\OrientationsWorkAroundQA"),
                                                photo_segmentation_detector=os.path.join(input_path,
                                                                                         r"Detectors\PhotoSegmentationDetector\CracksA_v1"),
                                                compute_line_width=True,
                                                remove_small_components=0.15,
                                                export_lines3DDGN=True,
                                                export_lines3DCesium=True,
                                                output=os.path.join(output_path, "L3D_BridgeWithCracks_CCOrientations_cloud")
                                                )

    # Classifying 3mx mesh from photo segmentation detector and detecting lines in space
    success, job_id = job_manager.addLines3DJob(job_name="L3D_BridgeWithCracks_3MX_cloud",
                                                process_on_cloud=True,
                                                itwin_id=itwin_id,
                                                meshes=os.path.join(input_path,
                                                                    r"DataSets\BridgeWithCracks\Mesh3MX"),
                                                oriented_photos=os.path.join(input_path,
                                                                             r"DataSets\BridgeWithCracks\OrientedPhotos"),
                                                photo_segmentation_detector=os.path.join(input_path,
                                                                                         r"Detectors\PhotoSegmentationDetector\CracksA_v1"),
                                                compute_line_width=True,
                                                remove_small_components=0.15,
                                                export_lines3DDGN=True,
                                                export_lines3DCesium=True,
                                                output=os.path.join(output_path, "L3D_BridgeWithCracks_3MX_cloud")
                                                )

    # Classifying 3sm mesh from photo segmentation detector and detecting lines in space
    success, job_id = job_manager.addLines3DJob(job_name="L3D_BridgeWithCracks_3SM_cloud",
                                                process_on_cloud=True,
                                                itwin_id=itwin_id,
                                                meshes=os.path.join(input_path,
                                                                    r"DataSets\BridgeWithCracks\Mesh3SM"),
                                                oriented_photos=os.path.join(input_path,
                                                                             r"DataSets\BridgeWithCracks\OrientedPhotos"),
                                                photo_segmentation_detector=os.path.join(input_path,
                                                                                         r"Detectors\PhotoSegmentationDetector\CracksA_v1"),
                                                compute_line_width=True,
                                                remove_small_components=0.15,
                                                export_lines3DDGN=True,
                                                export_lines3DCesium=True,
                                                output=os.path.join(output_path, "L3D_BridgeWithCracks_3SM_cloud")
                                                )

    # Classifying 3mx mesh from Applied AI (Onnx format) photo segmentation detector and detecting lines in space
    success, job_id = job_manager.addLines3DJob(job_name="L3D_BridgeWithCracks_3MX_ONNX_cloud",
                                                process_on_cloud=True,
                                                itwin_id=itwin_id,
                                                meshes=os.path.join(input_path,
                                                                    r"DataSets\BridgeWithCracks\Mesh3MX"),
                                                oriented_photos=os.path.join(input_path,
                                                                             r"DataSets\BridgeWithCracks\OrientedPhotos"),
                                                photo_segmentation_detector=os.path.join(input_path,
                                                                                         r"Detectors\PhotoSegmentationDetector\CracksAppliedAI_v1"),
                                                compute_line_width=True,
                                                remove_small_components=0.15,
                                                export_lines3DDGN=True,
                                                export_lines3DCesium=True,
                                                output=os.path.join(output_path, "L3D_BridgeWithCracks_3MX_ONNX_cloud")
                                                )

    # Given classified point clouds, detecting lines in space
    success, job_id = job_manager.addLines3DJob(job_name="L3D_spain_given_S3D_cloud",
                                                process_on_cloud=True,
                                                itwin_id=itwin_id,
                                                segmentation3D=os.path.join(input_path,
                                                                            r"GivenDetections\S3D_Spain"),
                                                compute_line_width=True,
                                                remove_small_components=0.15,
                                                export_srs="EPSG:32629",
                                                export_lines3DDGN=True,
                                                export_lines3DCesium=True,
                                                output=os.path.join(output_path, "L3D_spain_given_S3D_cloud")
                                                )

    success, job_id = job_manager.addLines3DJob(job_name="L3D_spain_given_S3D_OPC_cloud",
                                                process_on_cloud=True,
                                                itwin_id=itwin_id,
                                                segmentation3D=os.path.join(input_path,
                                                                            r"GivenDetections\S3D_Spain_OPC\segmentation3D"),
                                                compute_line_width=True,
                                                remove_small_components=0.15,
                                                export_srs="EPSG:32629",
                                                export_lines3DDGN=True,
                                                export_lines3DCesium=True,
                                                output=os.path.join(output_path, "L3D_spain_given_S3D_OPC_cloud")
                                                )

    # Given photo segmentation, classifying point clouds and detecting lines in space
    success, job_id = job_manager.addLines3DJob(job_name="L3D_BridgeWithCracks_POD_given_S2D_cloud",
                                                process_on_cloud=True,
                                                itwin_id=itwin_id,
                                                point_clouds=os.path.join(input_path,
                                                                          r"DataSets\BridgeWithCracks\PointClouds"),
                                                oriented_photos=os.path.join(input_path,
                                                                             r"DataSets\BridgeWithCracks\OrientedPhotos"),
                                                segmentation2D=os.path.join(input_path,
                                                                            r"GivenDetections\S2D_BridgeWithCracks"),
                                                compute_line_width=True,
                                                remove_small_components=0.15,
                                                export_lines3DDGN=True,
                                                export_lines3DCesium=True,
                                                output=os.path.join(output_path, "L3D_BridgeWithCracks_POD_given_S2D_cloud")
                                                )

    success, job_id = job_manager.addLines3DJob(job_name="L3D_BridgeWithCracks_OPC_given_S2D_cloud",
                                                process_on_cloud=True,
                                                itwin_id=itwin_id,
                                                point_clouds=os.path.join(input_path,
                                                                          r"DataSets\BridgeWithCracks\PointClouds_OPC"),
                                                oriented_photos=os.path.join(input_path,
                                                                             r"DataSets\BridgeWithCracks\OrientedPhotos"),
                                                segmentation2D=os.path.join(input_path,
                                                                            r"GivenDetections\S2D_BridgeWithCracks"),
                                                compute_line_width=True,
                                                remove_small_components=0.15,
                                                export_lines3DDGN=True,
                                                export_lines3DCesium=True,
                                                output=os.path.join(output_path, "L3D_BridgeWithCracks_OPC_given_S2D_cloud")
                                                )

    # Given photo segmentation, classifying 3MX mesh and detecting lines in space
    success, job_id = job_manager.addLines3DJob(job_name="L3D_BridgeWithCracks_3MX_given_S2D_cloud",
                                                process_on_cloud=True,
                                                itwin_id=itwin_id,
                                                meshes=os.path.join(input_path,
                                                                    r"DataSets\BridgeWithCracks\Mesh3MX"),
                                                oriented_photos=os.path.join(input_path,
                                                                             r"DataSets\BridgeWithCracks\OrientedPhotos"),
                                                segmentation2D=os.path.join(input_path,
                                                                            r"GivenDetections\S2D_BridgeWithCracks"),
                                                compute_line_width=True,
                                                remove_small_components=0.15,
                                                export_lines3DDGN=True,
                                                export_lines3DCesium=True,
                                                output=os.path.join(output_path, "L3D_BridgeWithCracks_3MX_given_S2D_cloud")
                                                )

    # Given photo segmentation, classifying 3SM mesh and detecting lines in space
    success, job_id = job_manager.addLines3DJob(job_name="L3D_BridgeWithCracks_3SM_given_S2D_cloud",
                                                process_on_cloud=True,
                                                itwin_id=itwin_id,
                                                meshes=os.path.join(input_path,
                                                                    r"DataSets\BridgeWithCracks\Mesh3SM"),
                                                oriented_photos=os.path.join(input_path,
                                                                             r"DataSets\BridgeWithCracks\OrientedPhotos"),
                                                segmentation2D=os.path.join(input_path,
                                                                            r"GivenDetections\S2D_BridgeWithCracks"),
                                                compute_line_width=True,
                                                remove_small_components=0.15,
                                                export_lines3DDGN=True,
                                                export_lines3DCesium=True,
                                                output=os.path.join(output_path, "L3D_BridgeWithCracks_3SM_given_S2D_cloud")
                                                )

    # ========================= Change Detection

    # Detecting changes between two point cloud collections
    success, job_id = job_manager.addChangeDetectionJob(job_name="Change_spain_cloud",
                                                        process_on_cloud=True,
                                                        itwin_id=itwin_id,
                                                        point_clouds1=os.path.join(input_path,
                                                                                   r"DataSets\SpainChange\PointClouds1"),
                                                        point_clouds2=os.path.join(input_path,
                                                                                   r"DataSets\SpainChange\PointClouds2"),
                                                        save_point_clouds=True,
                                                        color_threshold_low=0.3,
                                                        color_threshold_high=0.6,
                                                        min_points=50,
                                                        output=os.path.join(output_path, "Change_spain_cloud")
                                                        )

    success, job_id = job_manager.addChangeDetectionJob(job_name="Change_horn_local_cloud",
                                                        process_on_cloud=True,
                                                        itwin_id=itwin_id,
                                                        point_clouds1=os.path.join(input_path,
                                                                                   r"DataSets\Horn\NoSRS\PointClouds1"),
                                                        point_clouds2=os.path.join(input_path,
                                                                                   r"DataSets\Horn\NoSRS\PointClouds2"),
                                                        save_point_clouds=True,
                                                        dist_threshold_low=5,
                                                        dist_threshold_high=15,
                                                        min_points=1000,
                                                        output=os.path.join(output_path, "Change_horn_local_cloud")
                                                        )

    success, job_id = job_manager.addChangeDetectionJob(job_name="Change_horn_las_cloud",
                                                        process_on_cloud=True,
                                                        itwin_id=itwin_id,
                                                        point_clouds1=os.path.join(input_path,
                                                                                   r"DataSets\Horn\LAZ\PointClouds1"),
                                                        point_clouds2=os.path.join(input_path,
                                                                                   r"DataSets\Horn\LAZ\PointClouds2"),
                                                        save_point_clouds=True,
                                                        dist_threshold_low=5,
                                                        dist_threshold_high=15,
                                                        min_points=1000,
                                                        export_srs="EPSG:32633",
                                                        export_locations3DSHP=True,
                                                        output=os.path.join(output_path, "Change_horn_las_cloud")
                                                        )

    success, job_id = job_manager.addChangeDetectionJob(job_name="Change_horn_pod_cloud",
                                                        process_on_cloud=True,
                                                        itwin_id=itwin_id,
                                                        point_clouds1=os.path.join(input_path,
                                                                                   r"DataSets\Horn\POD\PointClouds1"),
                                                        point_clouds2=os.path.join(input_path,
                                                                                   r"DataSets\Horn\POD\PointClouds2"),
                                                        save_point_clouds=True,
                                                        dist_threshold_low=5,
                                                        dist_threshold_high=15,
                                                        min_points=1000,
                                                        export_srs="EPSG:32633",
                                                        export_locations3DSHP=True,
                                                        output=os.path.join(output_path, "Change_horn_pod_cloud")
                                                        )

    success, job_id = job_manager.addChangeDetectionJob(job_name="Change_horn_ply_cloud",
                                                        process_on_cloud=True,
                                                        itwin_id=itwin_id,
                                                        point_clouds1=os.path.join(input_path,
                                                                                   r"DataSets\Horn\PLY\PointClouds1"),
                                                        point_clouds2=os.path.join(input_path,
                                                                                   r"DataSets\Horn\PLY\PointClouds2"),
                                                        save_point_clouds=True,
                                                        dist_threshold_low=5,
                                                        dist_threshold_high=15,
                                                        min_points=1000,
                                                        output=os.path.join(output_path, "Change_horn_ply_cloud")
                                                        )

    success, job_id = job_manager.addChangeDetectionJob(job_name="Change_horn_opc_cloud",
                                                        process_on_cloud=True,
                                                        itwin_id=itwin_id,
                                                        point_clouds1=os.path.join(input_path,
                                                                                   r"DataSets\Horn\OPC\PointClouds1"),
                                                        point_clouds2=os.path.join(input_path,
                                                                                   r"DataSets\Horn\OPC\PointClouds2"),
                                                        save_point_clouds=True,
                                                        dist_threshold_low=5,
                                                        dist_threshold_high=15,
                                                        min_points=1000,
                                                        output=os.path.join(output_path, "Change_horn_opc_cloud")
                                                        )

    # Detecting changes between two mesh collections
    success, job_id = job_manager.addChangeDetectionJob(job_name="Change_lake_cloud",
                                                        process_on_cloud=True,
                                                        itwin_id=itwin_id,
                                                        meshes1=os.path.join(input_path,
                                                                             r"DataSets\Lake\Meshes1"),
                                                        meshes2=os.path.join(input_path,
                                                                             r"DataSets\Lake\Meshes2"),
                                                        save_point_clouds=True,
                                                        dist_threshold_low=5,
                                                        dist_threshold_high=15,
                                                        min_points=1000,
                                                        resolution=0.5,
                                                        output=os.path.join(output_path, "Change_lake_cloud")
                                                        )

# =============================  Monitoring ===========================
# Wait for on all jobs to complete
# Download cloud job results when they complete

job_manager.start_monitoring()
if not nocloud:
    ccmasterkernel.InsightStandaloneJob.endCloudProcessing()

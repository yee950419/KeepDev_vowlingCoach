/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================
*/

package org.tensorflow.lite.examples.poseestimation.camera

import android.annotation.SuppressLint
import android.content.ContentValues
import android.content.Context
import android.content.Intent
import android.content.pm.ActivityInfo
import android.graphics.Bitmap
import android.graphics.ImageFormat
import android.graphics.Matrix
import android.graphics.Rect
import android.hardware.camera2.*
import android.media.ImageReader
import android.media.MediaCodec
import android.media.MediaRecorder
import android.media.MediaScannerConnection
import android.os.Build
import android.os.Environment
import android.os.Handler
import android.os.HandlerThread
import android.provider.MediaStore
import android.util.Log
import android.util.Range
import android.view.Surface
import android.view.SurfaceView
import android.webkit.MimeTypeMap
import android.widget.ImageView
import android.widget.Toast
import androidx.core.content.FileProvider
import kotlinx.coroutines.suspendCancellableCoroutine
import org.tensorflow.lite.examples.poseestimation.BuildConfig
import org.tensorflow.lite.examples.poseestimation.R
import org.tensorflow.lite.examples.poseestimation.VisualizationUtils
import org.tensorflow.lite.examples.poseestimation.YuvToRgbConverter
import org.tensorflow.lite.examples.poseestimation.data.Person
import org.tensorflow.lite.examples.poseestimation.ml.MoveNetMultiPose
import org.tensorflow.lite.examples.poseestimation.ml.PoseClassifier
import org.tensorflow.lite.examples.poseestimation.ml.PoseDetector
import org.tensorflow.lite.examples.poseestimation.ml.TrackerType
import java.io.File
import java.io.IOException
import java.text.SimpleDateFormat
import java.util.*
import kotlin.coroutines.coroutineContext
import kotlin.coroutines.resume
import kotlin.coroutines.resumeWithException

class CameraSource(
    private val surfaceView: SurfaceView,
    private val listener: CameraSourceListener? = null
) {

    companion object {
        private const val PREVIEW_WIDTH = 640
        private const val PREVIEW_HEIGHT = 480

        /** Threshold for confidence score. */
        private const val MIN_CONFIDENCE = .2f
        private const val TAG = "Camera Source"

        private const val RECORDER_VIDEO_BITRATE: Int = 10_000_000
        private const val MIN_REQUIRED_RECORDING_TIME_MILLIS: Long = 1000L

        /** Creates a [File] named with the current date and time */
        private fun createFile(context: Context, extension: String): File {

            val sdf = SimpleDateFormat("yyyy_MM_dd_HH_mm_ss_SSS", Locale.US)
            return File(Environment.getExternalStoragePublicDirectory(
                Environment.DIRECTORY_DOWNLOADS), "VID_${sdf.format(Date())}.$extension")
//            return  File(context.getExternalFilesDir(null), "VID_${sdf.format(Date())}.$extension")
        }
    }



    private val lock = Any()
    private var detector: PoseDetector? = null
    private var classifier: PoseClassifier? = null
    private var isTrackerEnabled = false
    private var yuvConverter: YuvToRgbConverter = YuvToRgbConverter(surfaceView.context)
    private lateinit var imageBitmap: Bitmap

    /*
    비디오 녹화 코드
     */
    /** File where the recording will be saved */
    private var outputFile: File = createFile(surfaceView.rootView.context, "mp4")
    /**
     * Setup a persistent [Surface] for the recorder so we can use it as an output target for the
     * camera session without preparing the recorder
     */
    private val recorderSurface: Surface by lazy {

        // Get a persistent Surface from MediaCodec, don't forget to release when done
        val surface = MediaCodec.createPersistentInputSurface()

        // Prepare and release a dummy MediaRecorder with our new surface
        // Required to allocate an appropriately sized buffer before passing the Surface as the
        //  output target to the capture session
        createRecorder(surface).apply {
            prepare()
            release()
        }

        surface
    }

    /** Saves the video recording */
    private val recorder: MediaRecorder by lazy { createRecorder(recorderSurface) }


    //preview 캡쳐세션에 사용될 request들 -> 아마 이미 구현되어 있을듯
    /** Requests used for preview only in the [CameraCaptureSession] */
//    private val previewRequest: CaptureRequest by lazy {
//        // Capture request holds references to target surfaces
//        session.device.createCaptureRequest(CameraDevice.TEMPLATE_PREVIEW).apply {
//            // Add the preview surface target
//            addTarget(surfaceView.holder.surface)
//        }.build()
//    }

    /** Requests used for preview and recording in the [CameraCaptureSession] */
//    private val recordRequest: CaptureRequest by lazy {
//        // Capture request holds references to target surfaces
//        session.device.createCaptureRequest(CameraDevice.TEMPLATE_RECORD).apply {
//            // Add the preview and recording surface targets
//            addTarget(surfaceView.holder.surface)
//            addTarget(recorderSurface)
//            // Sets user requested FPS for all targets
//            set(CaptureRequest.CONTROL_AE_TARGET_FPS_RANGE, Range(framesPerSecond, framesPerSecond))
//        }.build()
//    }

    private var recordingStartMillis: Long = 0L

    /** Creates a [MediaRecorder] instance using the provided [Surface] as input */
    private fun createRecorder(surface: Surface) = MediaRecorder().apply {
        setAudioSource(MediaRecorder.AudioSource.MIC)
        setVideoSource(MediaRecorder.VideoSource.SURFACE)
        setOutputFormat(MediaRecorder.OutputFormat.MPEG_4)
        setVideoEncoder(MediaRecorder.VideoEncoder.H264)
        setAudioEncoder(MediaRecorder.AudioEncoder.AAC)
        setVideoSize(PREVIEW_WIDTH, PREVIEW_HEIGHT)
        if(framesPerSecond > 0) setVideoFrameRate(framesPerSecond)
        setOutputFile(outputFile.absolutePath)
        setVideoEncodingBitRate(RECORDER_VIDEO_BITRATE)
        setInputSurface(surface)
    }

    private var isRecording = false



    //원래 코드
    /** Frame count that have been processed so far in an one second interval to calculate FPS. */
    private var fpsTimer: Timer? = null
    private var frameProcessedInOneSecondInterval = 0
    private var framesPerSecond = 0

    /** Detects, characterizes, and connects to a CameraDevice (used for all camera operations) */
    private val cameraManager: CameraManager by lazy {
        val context = surfaceView.context
        context.getSystemService(Context.CAMERA_SERVICE) as CameraManager
    }

    /** Readers used as buffers for camera still shots */
    private var imageReader: ImageReader? = null

    /** The [CameraDevice] that will be opened in this fragment */
    private var camera: CameraDevice? = null
//    private lateinit var camera: CameraDevice
    /** Internal reference to the ongoing [CameraCaptureSession] configured with our parameters */
    private var session: CameraCaptureSession? = null
//    private lateinit var session: CameraCaptureSession

    /** [HandlerThread] where all buffer reading operations run */
    private var imageReaderThread: HandlerThread? = null

    /** [Handler] corresponding to [imageReaderThread] */
    private var imageReaderHandler: Handler? = null
    private var cameraId: String = ""

    suspend fun initCamera() {
        camera = openCamera(cameraManager, cameraId)

        imageReader =
            ImageReader.newInstance(PREVIEW_WIDTH, PREVIEW_HEIGHT, ImageFormat.YUV_420_888, 3)
        imageReader?.setOnImageAvailableListener({ reader ->
            val image = reader.acquireLatestImage()
            if (image != null) {
                if (!::imageBitmap.isInitialized) {
                    imageBitmap =
                        Bitmap.createBitmap(
                            PREVIEW_WIDTH,
                            PREVIEW_HEIGHT,
                            Bitmap.Config.ARGB_8888
                        )
                }
                yuvConverter.yuvToRgb(image, imageBitmap)
                // Create rotated version for portrait display
                val rotateMatrix = Matrix()
                rotateMatrix.postRotate(90.0f)

                val rotatedBitmap = Bitmap.createBitmap(
                    imageBitmap, 0, 0, PREVIEW_WIDTH, PREVIEW_HEIGHT,
                    rotateMatrix, false
                )
                processImage(rotatedBitmap)
                image.close()
            }
        }, imageReaderHandler)

        imageReader?.surface?.let { surface ->
            session = createSession(listOf(surface, recorderSurface))
            val cameraRequest = camera?.createCaptureRequest(
                CameraDevice.TEMPLATE_PREVIEW
            )?.apply {
                addTarget(surface)
            }
            cameraRequest?.build()?.let {
                session?.setRepeatingRequest(it, null, null)
            }

            val recordRequest = camera?.createCaptureRequest(
                CameraDevice.TEMPLATE_RECORD
            )?.apply {
                addTarget(surface)
                addTarget(recorderSurface)
//                set(CaptureRequest.CONTROL_AE_TARGET_FPS_RANGE, Range(framesPerSecond, framesPerSecond))
            }
            recordRequest?.build()?.let {
                session?.setRepeatingRequest(it, null, null)
            }

            val recordButton = surfaceView.rootView.findViewById<ImageView>(R.id.record_button)
            val stopButton = surfaceView.rootView.findViewById<ImageView>(R.id.stop_button)
            recordButton.setOnClickListener {
                if(isRecording) {
                    recorder.stop()
                    // Broadcasts the media file to the rest of the system
                    MediaScannerConnection.scanFile(
                        surfaceView.context, arrayOf(outputFile.absolutePath), null, null)

//                    // Launch external activity via intent to play video recorded using our provider
//                    surfaceView.context.startActivity(Intent().apply {
//                        action = Intent.ACTION_VIEW
//                        type = MimeTypeMap.getSingleton()
//                            .getMimeTypeFromExtension(outputFile.extension)
//                        val authority = "${BuildConfig.APPLICATION_ID}.provider"
//                        data = FileProvider.getUriForFile(surfaceView.context, authority, outputFile)
//                        flags = Intent.FLAG_GRANT_READ_URI_PERMISSION or
//                                Intent.FLAG_ACTIVITY_CLEAR_TOP
//                    })

                    recordButton.setImageResource(R.drawable.ic_record_btn)
                } else {
//                    requestedOrientation =
//                        ActivityInfo.SCREEN_ORIENTATION_LOCKED

                    recorder.prepare()
                    recorder.start()

                    recordingStartMillis = System.currentTimeMillis()
                    Log.d(TAG, "Recording started")

                    recordButton.setImageResource(R.drawable.ic_record_btn_red)
                }
                isRecording = !isRecording

//                showToast("$isRecording")

            }

        }


    }

//    private fun setUpMediaRecorder() {
//
//        if (nextVideoAbsolutePath.isNullOrEmpty()) {
//            nextVideoAbsolutePath = getVideoFilePath(surfaceView.rootView.context)
//        }
//
//        mediaRecorder?.apply {
//            setAudioSource(MediaRecorder.AudioSource.MIC)
//            setVideoSource(MediaRecorder.VideoSource.SURFACE)
//            setOutputFormat(MediaRecorder.OutputFormat.MPEG_4)
//            setOutputFile(nextVideoAbsolutePath)
//            setVideoEncodingBitRate(10000000)
//            setVideoFrameRate(30)
//            setVideoSize(PREVIEW_WIDTH, PREVIEW_HEIGHT)
//            setVideoEncoder(MediaRecorder.VideoEncoder.H264)
//            setAudioEncoder(MediaRecorder.AudioEncoder.AAC)
//            prepare()
//        }
//    }
//
//    private fun getVideoFilePath(context: Context?): String {
//        val filename = "${System.currentTimeMillis()}.mp4"
//        val dir = context?.getExternalFilesDir(null)
//
//        return if (dir == null) {
//            filename
//        } else {
//            "${dir.absolutePath}/$filename"
//        }
//
//    }
//
//    private fun startRecordingVideo() {
//        if(camera == null) return
//
//        try {
//            setUpMediaRecorder()
//
////            val texture = surfaceView.holder.surface.apply {
////                setDera
////            }
//
//            val previewSurface = surfaceView.holder.surface
//            val recorderSurface = mediaRecorder!!.surface
//            val surfaces = ArrayList<Surface>().apply {
//                add(previewSurface)
//                add(recorderSurface)
//            }
//
//            previewRequestBuilder = camera!!.createCaptureRequest(TEMPLATE_RECORD).apply {
//                addTarget(previewSurface)
//                addTarget(recorderSurface)
//            }
//
//            camera?.createCaptureSession(surfaces, object : CameraCaptureSession.StateCallback() {
//                override fun onConfigured(cameraCaptureSession: CameraCaptureSession) {
//                    session = cameraCaptureSession
//                    isRecording = true
//                    mediaRecorder?.start()
//                }
//
//                override fun onConfigureFailed(p0: CameraCaptureSession) {
//                    showToast("Failed")
//                }
//            }, imageReaderHandler)
//        } catch (e: CameraAccessException) {
//            Log.e(TAG, e.toString())
//        } catch (e: IOException) {
//            Log.e(TAG, e.toString())
//        }
//
//    }
//
//    private fun stopRecordingVideo() {
//        isRecording = false
//        mediaRecorder?.apply {
//            stop()
//            reset()
//        }
//
//
//    }



    private fun showToast(message: String) {
        Toast.makeText(surfaceView.context, message, Toast.LENGTH_LONG).show()
    }

    private suspend fun createSession(targets: List<Surface>): CameraCaptureSession =
        suspendCancellableCoroutine { cont ->
            camera?.createCaptureSession(targets, object : CameraCaptureSession.StateCallback() {
                override fun onConfigured(captureSession: CameraCaptureSession) =
                    cont.resume(captureSession)

                override fun onConfigureFailed(session: CameraCaptureSession) {
                    cont.resumeWithException(Exception("Session error"))
                }
            }, null)
        }

    @SuppressLint("MissingPermission")
    private suspend fun openCamera(manager: CameraManager, cameraId: String): CameraDevice =
        suspendCancellableCoroutine { cont ->


            manager.openCamera(cameraId, object : CameraDevice.StateCallback() {
                override fun onOpened(camera: CameraDevice) = cont.resume(camera)

                override fun onDisconnected(camera: CameraDevice) {
                    camera.close()
                }

                override fun onError(camera: CameraDevice, error: Int) {
                    if (cont.isActive) cont.resumeWithException(Exception("Camera error"))
                }
            }, imageReaderHandler)
        }

    fun prepareCamera() {
        for (cameraId in cameraManager.cameraIdList) {
            val characteristics = cameraManager.getCameraCharacteristics(cameraId)

            // We don't use a front facing camera in this sample.
            val cameraDirection = characteristics.get(CameraCharacteristics.LENS_FACING)
            if (cameraDirection != null &&
                cameraDirection == CameraCharacteristics.LENS_FACING_FRONT
            ) {
                continue
            }
            this.cameraId = cameraId
        }
    }

    fun setDetector(detector: PoseDetector) {
        synchronized(lock) {
            if (this.detector != null) {
                this.detector?.close()
                this.detector = null
            }
            this.detector = detector
        }
    }

    fun setClassifier(classifier: PoseClassifier?) {
        synchronized(lock) {
            if (this.classifier != null) {
                this.classifier?.close()
                this.classifier = null
            }
            this.classifier = classifier
        }
    }

    /**
     * Set Tracker for Movenet MuiltiPose model.
     */
    fun setTracker(trackerType: TrackerType) {
        isTrackerEnabled = trackerType != TrackerType.OFF
        (this.detector as? MoveNetMultiPose)?.setTracker(trackerType)
    }

    fun resume() {
        imageReaderThread = HandlerThread("imageReaderThread").apply { start() }
        imageReaderHandler = Handler(imageReaderThread!!.looper)
        fpsTimer = Timer()
        fpsTimer?.scheduleAtFixedRate(
            object : TimerTask() {
                override fun run() {
                    framesPerSecond = frameProcessedInOneSecondInterval
                    frameProcessedInOneSecondInterval = 0
                }
            },
            0,
            1000
        )
    }

    fun close() {
        session?.close()
        session = null
        camera?.close()
        camera = null
        imageReader?.close()
        imageReader = null
        stopImageReaderThread()
        detector?.close()
        detector = null
        classifier?.close()
        classifier = null
        fpsTimer?.cancel()
        fpsTimer = null
        frameProcessedInOneSecondInterval = 0
        framesPerSecond = 0
        recorder.release()
        recorderSurface.release()
    }

    // process image
    private fun processImage(bitmap: Bitmap) {
        val persons = mutableListOf<Person>()
        var classificationResult: List<Pair<String, Float>>? = null

        synchronized(lock) {
            detector?.estimatePoses(bitmap)?.let {
                persons.addAll(it)

                // if the model only returns one item, allow running the Pose classifier.
                if (persons.isNotEmpty()) {
                    classifier?.run {
                        classificationResult = classify(persons[0])
                    }
                }
            }
        }
        frameProcessedInOneSecondInterval++
        if (frameProcessedInOneSecondInterval == 1) {
            // send fps to view
            listener?.onFPSListener(framesPerSecond)
        }

        // if the model returns only one item, show that item's score.
        if (persons.isNotEmpty()) {
            listener?.onDetectedInfo(persons[0].score, classificationResult)
        }
        visualize(persons, bitmap)
    }

    private fun visualize(persons: List<Person>, bitmap: Bitmap) {

        val outputBitmap = VisualizationUtils.drawBodyKeypoints(
            bitmap,
            persons.filter { it.score > MIN_CONFIDENCE }, isTrackerEnabled
        )

        val holder = surfaceView.holder
        val surfaceCanvas = holder.lockCanvas()
        surfaceCanvas?.let { canvas ->
            val screenWidth: Int
            val screenHeight: Int
            val left: Int
            val top: Int

            if (canvas.height > canvas.width) {
                val ratio = outputBitmap.height.toFloat() / outputBitmap.width
                screenWidth = canvas.width
                left = 0
                screenHeight = (canvas.width * ratio).toInt()
                top = (canvas.height - screenHeight) / 2
            } else {
                val ratio = outputBitmap.width.toFloat() / outputBitmap.height
                screenHeight = canvas.height
                top = 0
                screenWidth = (canvas.height * ratio).toInt()
                left = (canvas.width - screenWidth) / 2
            }
            val right: Int = left + screenWidth
            val bottom: Int = top + screenHeight

            canvas.drawBitmap(
                outputBitmap, Rect(0, 0, outputBitmap.width, outputBitmap.height),
                Rect(left, top, right, bottom), null
            )
            surfaceView.holder.unlockCanvasAndPost(canvas)
        }
    }

    private fun stopImageReaderThread() {
        imageReaderThread?.quitSafely()
        try {
            imageReaderThread?.join()
            imageReaderThread = null
            imageReaderHandler = null
        } catch (e: InterruptedException) {
            Log.d(TAG, e.message.toString())
        }
    }

    interface CameraSourceListener {
        fun onFPSListener(fps: Int)

        fun onDetectedInfo(personScore: Float?, poseLabels: List<Pair<String, Float>>?)
    }
}

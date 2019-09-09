package com.kacper.cameraapp;

import androidx.appcompat.app.AppCompatActivity;

import android.bluetooth.BluetoothAdapter;
import android.bluetooth.BluetoothDevice;
import android.bluetooth.BluetoothSocket;
import android.os.Bundle;
import android.os.ParcelUuid;
import android.util.Log;
import android.view.SurfaceView;
import android.widget.Toast;

import org.opencv.android.CameraBridgeViewBase;
import org.opencv.android.JavaCameraView;
import org.opencv.android.OpenCVLoader;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfInt;
import org.opencv.core.MatOfPoint;
import org.opencv.core.MatOfPoint2f;
import org.opencv.core.Point;
import org.opencv.core.RotatedRect;
import org.opencv.core.Scalar;
import org.opencv.imgproc.Imgproc;
import org.opencv.imgproc.Moments;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.OutputStream;
import java.net.ServerSocket;
import java.net.Socket;
import java.util.ArrayList;
import java.util.List;
import java.util.Set;
import java.util.UUID;
import java.util.concurrent.ExecutionException;

public class MainActivity extends AppCompatActivity implements CameraBridgeViewBase.CvCameraViewListener2 {

    CameraBridgeViewBase cameraBridgeViewBase;
    Mat frame, hsv, mask, hierarchy, drawFrame, gray, extractedMask, extracted;

    Scalar lowerThresh;
    Scalar upperThresh;

    Scalar RED;
    Scalar GREEN;
    Scalar BLUE;
    Scalar YELLOW;
    Scalar WHITE;
    Scalar BLACK;

    Point labelLocation;

    String received;
    Thread serverThread;

    BluetoothAdapter btAdapter;
    BluetoothDevice hc05;
    BluetoothSocket hc05Socket;

    String address;
    String SERVICE_ID = "00001101-0000-1000-8000-00805f9b34fb"; //SPP UUID

    BTConnectThread btThread;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        if(OpenCVLoader.initDebug()){
            Toast.makeText(getApplicationContext(), "Loaded", Toast.LENGTH_LONG).show();
        }
        else{
            Toast.makeText(getApplicationContext(), "Not loaded", Toast.LENGTH_LONG).show();
        }

        cameraBridgeViewBase = (JavaCameraView)findViewById(R.id.myCameraView);
        cameraBridgeViewBase.setVisibility(SurfaceView.VISIBLE);
        cameraBridgeViewBase.setCvCameraViewListener(this);
        cameraBridgeViewBase.enableView();

        serverThread = new Thread(new ServerThread());
        serverThread.start();

        labelLocation = new Point(30, 100);

        btAdapter = BluetoothAdapter.getDefaultAdapter();
        if(btAdapter != null){
            if(!btAdapter.isEnabled()){
                btAdapter.enable();
            }
            Set<BluetoothDevice> devices = btAdapter.getBondedDevices();
            for(BluetoothDevice device: devices){
                if(device.getName().equals("HC-05")){
                    address = device.getAddress();
                }
            }
            hc05 = btAdapter.getRemoteDevice(address);
            btThread = new BTConnectThread(hc05);
            btThread.start();
        }

    }

    @Override
    public Mat onCameraFrame(CameraBridgeViewBase.CvCameraViewFrame inputFrame) {
        Log.e("Address", hc05.getAddress());
        String features;

        frame = inputFrame.rgba();
        Core.flip(frame, frame, 1);
        drawFrame = frame.clone();

        Imgproc.cvtColor(frame, hsv, Imgproc.COLOR_RGB2HSV);
        Core.inRange(hsv, lowerThresh, upperThresh, mask);

        List<MatOfPoint> contours = new ArrayList<>();
        Imgproc.findContours(mask, contours, hierarchy, Imgproc.RETR_TREE, Imgproc.CHAIN_APPROX_NONE);

        if(contours.size() > 0){
            //find the longest contour
            double maxArea = 0;
            List<MatOfPoint> maxContourList = new ArrayList<>();
            for(MatOfPoint contour:contours){
                double area = Imgproc.contourArea(contour);
                if(area > maxArea){
                    maxArea = area;
                    maxContourList.clear();
                    maxContourList.add(contour);
                }
            }
            MatOfPoint longestContour = maxContourList.get(0);

            //find the length of the longest contour
            MatOfPoint2f maxContour2f = new MatOfPoint2f();
            longestContour.convertTo(maxContour2f, CvType.CV_32F);
            double contourLength = Imgproc.arcLength(maxContour2f, true);

            //find convex hull
            MatOfInt convexHull = new MatOfInt();
            Imgproc.convexHull(longestContour, convexHull);

            //draw convex hull
            MatOfPoint hullPoints = hull2Points(convexHull, longestContour);
            List<MatOfPoint> hullPointsList = new ArrayList<>();
            hullPointsList.add(hullPoints);

            //find hull area and length
            double hullArea = Imgproc.contourArea(hullPoints);
            MatOfPoint2f hullPoints2f = new MatOfPoint2f();
            hullPoints.convertTo(hullPoints2f, CvType.CV_32F);
            double hullLength = Imgproc.arcLength(hullPoints2f, true);

            //find extreme points
            Point[] contourArray = longestContour.toArray();
            Point maxLeft = new Point(5000,0);
            Point maxRight = new Point(0,0);
            Point maxTop = new Point(0,5000);
            Point maxBottom = new Point(0,0);

            for(Point point: contourArray){
                double x = point.x;
                double y = point.y;

                if(x < maxLeft.x){
                    maxLeft = point;
                }
                if(x > maxRight.x){
                    maxRight = point;
                }
                if(y < maxTop.y){
                    maxTop = point;
                }
                if(y > maxBottom.y){
                    maxBottom = point;
                }
            }

            //find minAreaRect
            RotatedRect rectangle = Imgproc.minAreaRect(maxContour2f);
            double rotation = rectangle.angle;
            Point[] rectPoints = new Point[4];
            rectangle.points(rectPoints);

            //extract object
            Imgproc.cvtColor(frame, gray, Imgproc.COLOR_RGB2GRAY);
            extractedMask = gray.clone();
            extractedMask.setTo(BLACK);
            Imgproc.drawContours(extractedMask, maxContourList, 0, WHITE, -1);
            extracted.setTo(BLACK);
            gray.copyTo(extracted, extractedMask);

            //calculate moments
            Moments moments = Imgproc.moments(gray);

            //collect features
            features = moments.toString();
            features = features.concat(",");
            features = features.concat(Double.toString(maxArea));
            features = features.concat(",");
            features = features.concat(Double.toString(contourLength));
            features = features.concat(",");
            features = features.concat(Double.toString(hullArea));
            features = features.concat(",");
            features = features.concat(Double.toString(hullLength));
            features = features.concat(",");
            features = features.concat(Double.toString(rotation));
            features = features.concat(",");
            features = features.concat(Double.toString(maxLeft.x));
            features = features.concat(",");
            features = features.concat(Double.toString(Math.sqrt((maxLeft.x*maxLeft.x)+(maxLeft.y*maxLeft.y))));
            features = features.concat(",");
            features = features.concat(Double.toString(maxRight.x));
            features = features.concat(",");
            features = features.concat(Double.toString(Math.sqrt((maxRight.x*maxRight.x)+(maxRight.y*maxRight.y))));
            features = features.concat(",");
            features = features.concat(Double.toString(maxTop.y));
            features = features.concat(",");
            features = features.concat(Double.toString(Math.sqrt((maxTop.x*maxTop.x)+(maxTop.y*maxTop.y))));
            features = features.concat(",");
            features = features.concat(Double.toString(maxBottom.y));
            features = features.concat(",");
            features = features.concat(Double.toString(Math.sqrt((maxBottom.x*maxBottom.x)+(maxBottom.y*maxBottom.y))));

            MessageSender messageSender = new MessageSender();
            try {
                messageSender.execute(features).get();
            } catch (ExecutionException e) {
                e.printStackTrace();
            } catch (InterruptedException e) {
                e.printStackTrace();
            }

            if(received != null){
                Imgproc.putText(drawFrame, received, labelLocation, 1, 5, RED, 10);
                if(hc05Socket != null){
                    try {
                        OutputStream out = hc05Socket.getOutputStream();
                        out.write((received + "\r\n").getBytes());
                    } catch (IOException e) {
                        Log.e("BT output error", "Can't create output stream");
                    }
                }
            }
            Imgproc.drawContours(drawFrame, maxContourList, -1, BLUE, 5);
            Imgproc.drawContours(drawFrame, hullPointsList, 0, GREEN, 5);
            Imgproc.circle(drawFrame, maxLeft, 10, RED, -1);
            Imgproc.circle(drawFrame, maxRight, 10, RED, -1);
            Imgproc.circle(drawFrame, maxTop, 10, RED, -1);
            Imgproc.circle(drawFrame, maxBottom, 10, RED, -1);
            for(int i = 0; i < 4; ++i){
                Imgproc.line(drawFrame, rectPoints[i], rectPoints[(i+1)%4], YELLOW, 5);
            }

            return drawFrame;
        }

        return frame;
    }

    @Override
    public void onCameraViewStarted(int width, int height) {
        lowerThresh = new Scalar(80, 30, 30);
        upperThresh = new Scalar(140, 255, 255);
        RED = new Scalar(255, 0, 0);
        GREEN = new Scalar(0, 255, 0);
        BLUE = new Scalar(0, 0, 255);
        YELLOW = new Scalar(255, 255, 0);
        WHITE = new Scalar(255, 255, 255);
        BLACK = new Scalar(0,0,0);

        frame = new Mat(width, height, CvType.CV_8UC4);
        hsv = new Mat(width, height, CvType.CV_8UC4);
        mask = new Mat(width, height, CvType.CV_8UC4);
        drawFrame = new Mat(width, height, CvType.CV_8UC4);
        gray = new Mat(width, height, CvType.CV_8UC4);
        hierarchy = new Mat();
        extracted = new Mat();
    }

    @Override
    public void onCameraViewStopped() {
        frame.release();
        hsv.release();
        mask.release();
        drawFrame.release();
        hierarchy.release();
        gray.release();
        extractedMask.release();
        extracted.release();
    }

    @Override
    protected void onPause() {
        super.onPause();
        if(cameraBridgeViewBase!=null){
            cameraBridgeViewBase.disableView();
        }
    }

    @Override
    protected void onResume() {
        super.onResume();
        cameraBridgeViewBase.enableView();
        if(!OpenCVLoader.initDebug()){
            Toast.makeText(getApplicationContext(), "There's a problem with OpenCV!", Toast.LENGTH_LONG).show();
        }
        else{
            Toast.makeText(getApplicationContext(), "Resumed without errors", Toast.LENGTH_LONG).show();
        }
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        if(cameraBridgeViewBase!=null){
            cameraBridgeViewBase.disableView();
        }
    }

    MatOfPoint hull2Points(MatOfInt hull, MatOfPoint contour) {
        List<Integer> indexes = hull.toList();
        List<Point> points = new ArrayList<>();
        MatOfPoint point= new MatOfPoint();
        for(Integer index:indexes) {
            points.add(contour.toList().get(index));
        }
        point.fromList(points);
        return point;
    }

    class ServerThread implements Runnable {
        ServerSocket ss;
        Socket s;
        InputStreamReader isr;
        BufferedReader br;

        @Override
        public void run() {
            try {
                ss = new ServerSocket(7777);
                while (true){
                    s = ss.accept();
                    isr = new InputStreamReader(s.getInputStream());
                    br = new BufferedReader(isr);
                    received = br.readLine();
                    br.close();
                    isr.close();
                    s.close();
                }
            } catch (IOException e) {
                Log.w("Server socket error", "Error while creating server socket");
            }
        }
    }

    class BTConnectThread extends Thread {
        private final BluetoothSocket thisSocket;
        private final BluetoothDevice thisDevice;

        public BTConnectThread(BluetoothDevice device) {
            BluetoothSocket tmp = null;
            thisDevice = device;

            try {
                tmp = thisDevice.createRfcommSocketToServiceRecord(UUID.fromString(SERVICE_ID));
            } catch (IOException e) {
                Log.e("TEST", "Can't connect to service");
            }
            thisSocket = tmp;
        }

        public void run() {
            // Cancel discovery because it otherwise slows down the connection.
            btAdapter.cancelDiscovery();

            try {
                thisSocket.connect();
                Log.d("TESTING", "Connected to shit");
            } catch (IOException connectException) {
                try {
                    thisSocket.close();
                } catch (IOException closeException) {
                    Log.e("TEST", "Can't close socket");
                }
                return;
            }

            hc05Socket = thisSocket;

        }
        public void cancel() {
            try {
                thisSocket.close();
            } catch (IOException e) {
                Log.e("TEST", "Can't close socket");
            }
        }
    }
}

package com.zh.camera;

import org.opencv.android.CameraBridgeViewBase;
import org.opencv.android.Utils;
import org.opencv.core.Mat;
import org.opencv.core.Point;
import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.Matrix;
import android.graphics.Paint;
import android.graphics.Rect;
import android.util.AttributeSet;
import android.util.Log;

public abstract class BaseView extends CameraBridgeViewBase{

	long time;
	Rect storeFaceRect = new Rect(-1, -1, -1, -1);
	Point storeLeftEyeL = new Point(-1, -1);
	Point storeRightEye = new Point(-1, -1);
	float degrees;
	  
	public void setDraw(org.opencv.core.Rect storeFaceRect, Point storeLeftEyeL, Point storeRightEye){
		this.storeFaceRect.left = storeFaceRect.x +16;
		this.storeFaceRect.right = storeFaceRect.x + storeFaceRect.width+16;
		this.storeFaceRect.top = storeFaceRect.y-16;
		this.storeFaceRect.bottom = storeFaceRect.y + storeFaceRect.height-16;
		this.storeLeftEyeL = storeLeftEyeL;
		this.storeRightEye = storeRightEye;
	}
	
	public void setRotate(float degrees){
		this.degrees = degrees;
	}
	
	
	public BaseView(Context context, AttributeSet attrs) {
		super(context, attrs);
	}
	
	public BaseView(Context context, int cameraId) {
		super(context, cameraId);
	}
	@Override
	protected void deliverAndDrawFrame(CvCameraViewFrame frame) {
        Mat modified;
       
        if (mListener != null) {
            modified = mListener.onCameraFrame(frame);
        } else {
            modified = frame.rgba();
        }
        
        boolean bmpValid = true;
        if (modified != null) {
            try {
            	mCacheBitmap = Bitmap.createBitmap(modified.cols(), modified.rows(), Bitmap.Config.ARGB_8888);
                Utils.matToBitmap(modified, mCacheBitmap);
            } catch(Exception e) {
                Log.e(TAG, "Mat type: " + modified);
                Log.e(TAG, "Bitmap type: " + mCacheBitmap.getWidth() + "*" + mCacheBitmap.getHeight());
                Log.e(TAG, "Utils.matToBitmap() throws an exception: " + e.getMessage());
                bmpValid = false;
            }
        }
        
        if (bmpValid && mCacheBitmap != null) {
            Canvas canvas = getHolder().lockCanvas();
            if (canvas != null) {
               canvas.drawColor(0, android.graphics.PorterDuff.Mode.CLEAR);
               Log.d(TAG, "mStretch value: " + mScale);
               if(degrees == 0f){
                if (mScale != 0) {
                    canvas.drawBitmap(mCacheBitmap, new Rect(0,0,mCacheBitmap.getWidth(), mCacheBitmap.getHeight()),
                         new Rect((int)((canvas.getWidth() - mScale*mCacheBitmap.getWidth()) / 2),
                         (int)((canvas.getHeight() - mScale*mCacheBitmap.getHeight()) / 2),
                         (int)((canvas.getWidth() - mScale*mCacheBitmap.getWidth()) / 2 + mScale*mCacheBitmap.getWidth()),
                         (int)((canvas.getHeight() - mScale*mCacheBitmap.getHeight()) / 2 + mScale*mCacheBitmap.getHeight())), null);
                } else {
                     canvas.drawBitmap(mCacheBitmap, new Rect(0,0,mCacheBitmap.getWidth(), mCacheBitmap.getHeight()),
                         new Rect((canvas.getWidth() - mCacheBitmap.getWidth()) / 2,
                         (canvas.getHeight() - mCacheBitmap.getHeight()) / 2,
                         (canvas.getWidth() - mCacheBitmap.getWidth()) / 2 + mCacheBitmap.getWidth(),
                         (canvas.getHeight() - mCacheBitmap.getHeight()) / 2 + mCacheBitmap.getHeight()), null);
                }
               }else{
            	   Matrix matrix = new Matrix();
                   
                   matrix.preTranslate((canvas.getWidth() - mCacheBitmap.getWidth()) / 2,(canvas.getHeight() - mCacheBitmap.getHeight()) / 2);
                   
                    matrix.postRotate(degrees,(canvas.getWidth()) / 2,(canvas.getHeight()) / 2);// 90f
                   
                   canvas.drawBitmap(mCacheBitmap, matrix, new Paint());
               }
            
            if (mFpsMeter != null) {
                mFpsMeter.measure();
                mFpsMeter.draw(canvas, 20, 30);
            }
          //  Log.e("bit", "w:"+mCacheBitmap.getWidth()+" h:"+mCacheBitmap.getHeight());
            getHolder().unlockCanvasAndPost(canvas);
            }
        }
    }
}

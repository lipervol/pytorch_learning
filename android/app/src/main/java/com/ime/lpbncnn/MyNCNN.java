package com.ime.lpbncnn;

import android.content.res.AssetManager;
import android.graphics.Bitmap;

public class MyNCNN {
    static {
        System.loadLibrary("lpbncnn");
    }

    public native boolean Init(AssetManager mgr);

    public native String Detect(Bitmap bitmap);
}

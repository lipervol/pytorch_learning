package com.ime.lpbncnn;

import androidx.annotation.Nullable;
import androidx.appcompat.app.AppCompatActivity;

import android.content.Intent;
import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.BitmapRegionDecoder;
import android.net.Uri;
import android.os.Bundle;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;

import com.ime.lpbncnn.databinding.ActivityMainBinding;

import java.io.FileNotFoundException;

public class MainActivity extends AppCompatActivity {

    private MyNCNN lpbncnn = new MyNCNN();
    private Button button1;
    private Button button2;
    private ImageView imageView;
    private TextView textView;
    private Bitmap selectedImage = null;
    private static final int SELECT_IMAGE = 1;

    @Override
    protected void onActivityResult(int requestCode, int resultCode, @Nullable Intent data) {
        super.onActivityResult(requestCode, resultCode, data);

        if(resultCode == RESULT_OK&&null != data){
            Uri selected = data.getData();

            try {
                if(requestCode == SELECT_IMAGE){
                    Bitmap bitmap = decodeUri(selected,32);
                    Bitmap rgba = bitmap.copy(Bitmap.Config.ARGB_8888,true);

                    selectedImage = Bitmap.createScaledBitmap(rgba,32,32,true);
                    rgba.recycle();
                    imageView.setImageBitmap(decodeUri(selected,400));
                }
            } catch (FileNotFoundException e) {
                e.printStackTrace();
            }
        }

    }

    private Bitmap decodeUri(Uri selected,int size) throws FileNotFoundException {
        BitmapFactory.Options o = new BitmapFactory.Options();
        o.inJustDecodeBounds = true;
        BitmapFactory.decodeStream(getContentResolver().openInputStream(selected),null,o);

        int width_tmp = o.outWidth,height=o.outHeight;
        int scale = 1;

        while(true){
            if(width_tmp/2 < size || height/2 < size) break;
            width_tmp /= 2;
            height /= 2;
            scale *= 2;
        }

        BitmapFactory.Options o2 = new BitmapFactory.Options();
        o2.inSampleSize = scale;
        return BitmapFactory.decodeStream(getContentResolver().openInputStream(selected),null,o2);
    }

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        button1 = findViewById(R.id.button);
        button2 = findViewById(R.id.button2);
        imageView = findViewById(R.id.imageView);
        textView = findViewById(R.id.textView);
        boolean ret=lpbncnn.Init(getAssets());

        if(!ret) Log.e("lpb", "NCNN_Init_Failed!");

        button1.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                Intent intent = new Intent(Intent.ACTION_PICK);
                intent.setType("image/*");
                startActivityForResult(intent,SELECT_IMAGE);
            }
        });

        button2.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                if(selectedImage == null) return;

                String result = lpbncnn.Detect(selectedImage);
                if(result == null){
                    textView.setText("识别失败");
                }
                else{
                    textView.setText(result);
                }
            }
        });

    }
}
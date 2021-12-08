#include <jni.h>
#include <string>
#include <vector>

#include <android/asset_manager_jni.h>
#include <android/bitmap.h>

#include <net.h>
#include <benchmark.h>

static ncnn::UnlockedPoolAllocator g_blob_pool_allocator;
static ncnn::PoolAllocator g_workspace_pool_allocator;

static std::vector<std::string> output_words;
static ncnn::Net mynet;

static std::vector<std::string> split_string(const std::string& str,const std::string& delimiter){

    std::vector<std::string> strings;

    std::string::size_type pos=0;
    std::string::size_type prev=0;

    while((pos=str.find(delimiter,prev))!=std::string::npos){
        strings.push_back(str.substr(prev,pos-prev));
        prev=pos+1;
    }

    strings.push_back(str.substr(prev));

    return strings;

}

extern "C"
{
    JNIEXPORT jboolean JNICALL Java_com_ime_lpbncnn_MyNCNN_Init(JNIEnv* env,jobject thiz,jobject assetManager){

        ncnn::Option opt;
        opt.lightmode=true;
        opt.num_threads=8;
        opt.blob_allocator=&g_blob_pool_allocator;
        opt.workspace_allocator=&g_workspace_pool_allocator;

        if(ncnn::get_gpu_count()!=0){
            opt.use_vulkan_compute=true;
        }

        AAssetManager* mgr=AAssetManager_fromJava(env,assetManager);

        mynet.opt=opt;

        {
            int ret = mynet.load_param(mgr, "resnet.param");
            if (ret != 0) {
                return JNI_FALSE;
            }
        }

        {
            int ret = mynet.load_model(mgr, "resnet.bin");
            if (ret != 0) {
                return JNI_FALSE;
            }
        }

        {
            AAsset* asset=AAssetManager_open(mgr,"resnet_words.txt",AASSET_MODE_BUFFER);
            if(!asset){
                return JNI_FALSE;
            }

            int len=AAsset_getLength(asset);
            std::string words_buffer;
            words_buffer.resize(len);
            int ret=AAsset_read(asset,(void*)words_buffer.data(),len);
            AAsset_close(asset);

            if(ret!=len){
                return JNI_FALSE;
            }

            output_words=split_string(words_buffer,"\n");
        }

        return JNI_TRUE;

    }

    JNIEXPORT jstring JNICALL Java_com_ime_lpbncnn_MyNCNN_Detect(JNIEnv* env,jobject thiz,jobject bitmap){
        if(ncnn::get_gpu_count()==0){
            return env->NewStringUTF("No Available GPU!");
        }

        AndroidBitmapInfo  info;
        AndroidBitmap_getInfo(env,bitmap,&info);
        int width=info.width;
        int height=info.height;
        if(width!=32||height!=32){
            return NULL;
        }
        if(info.format!=ANDROID_BITMAP_FORMAT_RGBA_8888){
            return NULL;
        }

        ncnn::Mat input=ncnn::Mat::from_android_bitmap(env,bitmap,ncnn::Mat::PIXEL_RGB);
        std::vector<float> class_score;
        {
            const float norm_vals[3]={1.0/255,1.0/255,1.0/255};
            input.substract_mean_normalize(0,norm_vals);

            ncnn::Extractor ex=mynet.create_extractor();

            ex.set_vulkan_compute(JNI_TRUE);

            const int in = 0;
            const int out = 54;

            ex.input(in,input);

            ncnn::Mat output;
            ex.extract(out,output);

            class_score.resize(output.w);
            for(int i=0;i<output.w;i++){
                class_score[i]=output[i];
            }

        }

        int top_class=0;
        float max_score=0.f;
        for(int i=0;i<class_score.size();i++){
            float s=class_score[i];
            if(s>max_score){
                top_class=i;
                max_score=s;
            }
        }

        const std::string& word=output_words[top_class];
        char score[32];
        sprintf(score,"%.3f",max_score);
        std::string result_str=" 识别结果："+std::string(word.c_str())+"\n 输出："+score;

        jstring result=env->NewStringUTF(result_str.c_str());

        return result;

    }

}
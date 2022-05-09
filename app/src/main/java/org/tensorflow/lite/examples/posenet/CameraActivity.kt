/*
 * Copyright 2019 The TensorFlow Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *       http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.tensorflow.lite.examples.posenet

import android.content.pm.PackageManager
import android.os.Build
import android.os.Bundle
import android.speech.tts.TextToSpeech
import android.util.Base64
import android.util.Log
import android.view.WindowManager
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import java.security.MessageDigest
import java.util.*


class CameraActivity : AppCompatActivity() {

  var ttsClient : TextToSpeech? = null

  override fun onCreate(savedInstanceState: Bundle?) {
    super.onCreate(savedInstanceState)
    setContentView(R.layout.tfe_pn_activity_camera)
    getAppKeyHash()
    window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
    savedInstanceState ?: supportFragmentManager.beginTransaction()
      .replace(R.id.container, PosenetActivity())
      .commit()
  }

  private fun getAppKeyHash() {
    try {
      val info = packageManager.getPackageInfo(packageName, PackageManager.GET_SIGNATURES)
      for (signature in info.signatures) {
        var md: MessageDigest
        md = MessageDigest.getInstance("SHA")
        md.update(signature.toByteArray())
        val something: String = String(Base64.encode(md.digest(), 0))
        Log.e("Hash key", something)
      }
    } catch (e: Exception) {
      // TODO Auto-generated catch block
      Log.e("name not found", e.toString())
    }
  }

  private fun initTextToSpeech() {
    if (Build.VERSION.SDK_INT < Build.VERSION_CODES.LOLLIPOP) {
      //Toast.makeText("SDK version is low", Toast.LENGTH_SHORT).show()
    }

    ttsClient = TextToSpeech(this) {
      if(it == TextToSpeech.SUCCESS) {
        var result = ttsClient?.setLanguage(Locale.KOREAN)
        if(result == TextToSpeech.LANG_MISSING_DATA || result == TextToSpeech.LANG_NOT_SUPPORTED) {
          Toast.makeText(this, "Language not supported", Toast.LENGTH_SHORT).show()
        }
        Toast.makeText(this, "TTS setting seccessed", Toast.LENGTH_SHORT).show()
      } else {
        Toast.makeText(this, "TTS init failed", Toast.LENGTH_SHORT).show()
      }
    }
  }

  private fun ttsSpeak(strTTS: String) {
    ttsClient?.speak(strTTS, TextToSpeech.QUEUE_ADD, null, null)
  }


}

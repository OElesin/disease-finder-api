<template>
  <div class="">
<el-menu
  :default-active="activeIndex2"
  class="el-menu-demo"
  mode="horizontal"
  @select="handleSelect"
  background-color="#545c64"
  text-color="#fff"
  active-text-color="#ffd04b">
  <el-menu-item index="1">Home</el-menu-item>
  </el-menu>

<br/>

    <el-upload
      action="#"
      list-type="picture-card"
      :on-preview="handlePictureCardPreview"
      :auto-upload="false"
      :multiple="true"
      :on-change="classify"
      :on-remove="handleRemove"
    >
      <i class="el-icon-upload2"></i>
    </el-upload>
    <el-dialog :visible.sync="dialogVisible">
      <img width="100%" :src="dialogImageUrl" alt />
    </el-dialog>

<br/>
<el-steps :active="active" finish-status="success">
  <el-step title="Upload Images"></el-step>
  <el-step title="Classify Images"></el-step>
  <el-step title="See Results"></el-step>
</el-steps>

<el-row>
  <el-col v-bind:body-style="{ padding: '10px' }" :span="6" v-for="(file, index) in files"  :key="file" :offset="index > 0 ? 0 : 0">
    <el-card :body-style="{ margin: '10px'}" >
      <img  :src="file.url">
        <div style="padding: 14px;">
        <h2>{{file.prediction}}</h2>
        <div class="bottom clearfix">
              <div>
                    <span v-bind:class="[file.status ? 'dot dot-success': 'dot dot-error']">
                      </span>
                  <p> {{file.status? 'Healthy':'Infected'}}</p>
              </div>
              
        </div>
      </div>
    </el-card>
  </el-col>
</el-row>

  </div>
</template>
<script>
import ImageLoader from "../plugins/imageloader";
import { InferenceSession, Tensor } from "onnxjs";
import getClass from "../plugins/classes";

export default {
  data() {
    return {
      dialogImageUrl: "",
      dialogVisible: false,
      files: [],
      classified:[],
    };
  },
  methods: {
    handleRemove(file, fileList) {
      console.log(file, fileList);
    },
    handlePictureCardPreview(file) {
      this.dialogImageUrl = file.url;
      this.dialogVisible = true;
    },
    async classify(file) {
      const url = file.url;
      const imageSize = 224;
      const session = new InferenceSession();
      await session.loadModel("/torch_model.onnx");
      const imageLoader = new ImageLoader(imageSize, imageSize);
      const imageData = await imageLoader.getImageData(url);

      const width = imageSize;
      const height = imageSize;
      const preprocessedData = imageLoader.preprocess(
        imageData.data,
        width,
        height
      );
      const inputTensor = new Tensor(preprocessedData, "float32", [
        1,
        3,
        width,
        height
      ]);
      // Run model with Tensor inputs and get the result.
      const outputMap = await session.run([inputTensor]);
      const outputData = outputMap.values().next().value.data;
      const prediction = this.printMatches(outputData);
      const status = prediction.includes("healthy");
      const item  = { status, prediction , url };
      if(!this.files.includes(item)) {
        this.files.push(item)
      }
    },
    printMatches(data) {
      let outputClasses = [];
      if (!data || data.length === 0) {
        const empty = [];
        for (let i = 0; i < 5; i++) {
          empty.push({ name: "-", probability: 0, index: 0 });
        }
        outputClasses = empty;
      } else {
        outputClasses = getClass(data, 1);
      }

      return outputClasses[0];
    }
  }
};
</script>

<style scoped>
.success{
  border:1px solid springgreen
}
.error {
  border: 1px solid red
}
.dot {
  height: 25px;
  width: 25px;
  background-color: #bbb;
  border-radius: 50%;
  display: inline-block;
}
.dot-error{
  background-color:red
}
.dot-success {
background-color:springgreen

}
</style>
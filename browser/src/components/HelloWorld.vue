<template>
  <div>
    <el-upload
      action="#"
      list-type="picture-card"
      :auto-upload="false"
      :on-change="classify"
      :on-preview="handlePictureCardPreview"
      :on-remove="handleRemove"
    >
      <i class="el-icon-plus"></i>
    </el-upload>
    <el-dialog :visible.sync="dialogVisible">
      <img width="100%" :src="dialogImageUrl" alt />
    </el-dialog>
  </div>
</template>
<script>
import ImageLoader from "../plugins/imageloader";
import { InferenceSession, Tensor } from "onnxjs";

export default {
  data() {
    return {
      dialogImageUrl: "",
      dialogVisible: false
    };
  },
  methods: {
    handleRemove(file, fileList) {
      console.log(file, fileList);
    },
    handlePictureCardPreview(file) {
      console.log(file)
      this.dialogImageUrl = file.url;
      console.log(file.url)
      this.dialogVisible = true;
    },
    async classify(file) {
      console.log(file)
      const url = file.url;
      const imageSize = 124;
      const session = new InferenceSession({ backendHint: "webgl" });
      await session.loadModel("/torch_model.onnx");
      const imageLoader = new ImageLoader(imageSize, imageSize);
      const imageData = await imageLoader.getImageData(url);
      console.log(imageData);

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
      console.log(outputMap);
      const outputData = outputMap.values().next().value.data;
      console.log(outputData)
    }
  }
};
</script>

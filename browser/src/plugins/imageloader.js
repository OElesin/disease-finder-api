/**
 * Loads an image from url
 */
import ndarray from "ndarray";
import ops  from "ndarray-ops"

class ImageLoader {
  constructor(imageWidth, imageHeight) {
    this.canvas = document.createElement("canvas");
    this.canvas.width = imageWidth;
    this.canvas.height = imageHeight;
    this.ctx = this.canvas.getContext("2d");
  }
  async getImageData(url) {
    await this.loadImageAsync(url);
    const imageData = this.ctx.getImageData(
      0,
      0,
      this.canvas.width,
      this.canvas.height
    );
    return imageData;
  }
  loadImageAsync(url) {
    return new Promise(resolve => {
      this.loadImageCb(url, () => {
        resolve();
      });
    });
  }
  loadImageCb(url, cb) {
    const img = new Image(); // Create new img element
    img.src = url;
    // load image data onto input canvas
    this.ctx.drawImage(img, 0, 0);
    //console.log(`image was loaded`);
    window.setTimeout(() => {
      cb();
    }, 0);
  }
  preprocess(data, width, height) {
    const dataFromImage = ndarray(new Float32Array(data), [width, height, 4]);
    const dataProcessed = ndarray(new Float32Array(width * height * 3), [
      1,
      3,
      height,
      width
    ]);

    // Normalize 0-255 to (-1)-1
    ops.divseq(dataFromImage, 128.0);
    ops.subseq(dataFromImage, 1.0);

    // Realign imageData from [224*224*4] to the correct dimension [1*3*224*224].
    ops.assign(
      dataProcessed.pick(0, 0, null, null),
      dataFromImage.pick(null, null, 2)
    );
    ops.assign(
      dataProcessed.pick(0, 1, null, null),
      dataFromImage.pick(null, null, 1)
    );
    ops.assign(
      dataProcessed.pick(0, 2, null, null),
      dataFromImage.pick(null, null, 0)
    );

    return dataProcessed.data;
  }
}
export default ImageLoader;

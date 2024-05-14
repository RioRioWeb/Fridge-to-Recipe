// window.onload = function () {
//   document.getElementById('file-sample').addEventListener('change', function (e) {
//       var file = e.target.files[0];
//       var blobUrl = window.URL.createObjectURL(file); // 選択された画像の一時的なURL
//       var img = document.getElementById('file-preview');
//       img.src = blobUrl;
//   });
// }

window.onload = function () {
  document.getElementById('file-sample').addEventListener('change', function (e) {
    var file = e.target.files[0];
    var reader = new FileReader();
    reader.onload = function (e) {
      var img = document.createElement('img');
      img.src = e.target.result;
      img.style.height = '150px';
      var displayDiv = document.querySelector('.centered.img-display');
      // まず表示エリアをクリア
      displayDiv.innerHTML = '';
      // 画像を表示エリアに追加
      displayDiv.appendChild(img);
    };
    reader.readAsDataURL(file);
  });
}

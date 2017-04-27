var app = angular.module('app', []);

app.controller("IndexCtrl", function($http) {
	var vm = this;

	vm.hello = "Hello!";

	vm.loadImage = function() {
		var file    = document.getElementById('upload-element').files[0];
		var reader  = new FileReader();

		reader.addEventListener("load", function () {
			vm.imageData = reader.result;
		}, false);

		reader.readAsDataURL(file);
	};
	vm.uploadImage = function() {
		vm.loading = true;

		var formData = new FormData();
		var fileElement = document.getElementById("upload-element");
		formData.append("file", fileElement.files[0]);

		$http({
			url: "/api/classify",
			method: "POST",
			data: formData,
			headers: {'Content-Type': undefined}
		}).success(function (response) {
			vm.response = response;
			console.log(response);
			vm.loading = false;
		});
	};
	vm.reset = function() {
		vm.loading = false;
		vm.response = null;
		vm.imageData = null;
	};

	document.getElementById("upload-element").addEventListener("change", vm.loadImage);
	vm.reset();
});

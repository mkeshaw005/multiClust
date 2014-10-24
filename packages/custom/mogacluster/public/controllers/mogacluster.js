'use strict';

angular.module('mean.mogacluster').controller('MogaclusterController', ['$scope', '$stateParams', '$location', 'Global', 'Mogacluster',
  function($scope, $stateParams, $location, Global, Mogacluster) {
    $scope.global = Global;

	  $scope.mutationRate = 0.01;
	  $scope.numIterations = 100;
	  $scope.populationSize = 50;
	  $scope.isCollapsed = false;
		$scope.package = {
      name: 'mogacluster'
    };
	  $scope.hasAuthorization = function(mogaCluster) {
		  if (!mogaCluster || !mogaCluster.user) return false;
		  return $scope.global.isAdmin || mogaCluster.user._id === $scope.global.user._id;
	  };

		$scope.openCreateView = function() {
			window.open('/#!/mogacluster/create', '_blank');
		};

	  $scope.create = function() {
		  var paramObject = {
			  title: this.title,
			  analysisParameters: {
				  foldChange: this.clusteringParams.generalOptions.foldChange.$modelValue,
				  experimentsOverFoldChange: this.clusteringParams.generalOptions.experimentsOverFoldChange.$modelValue,
				  dataFileName: this.dataFileName,
				  mutationRate: this.clusteringParams.advancedOptions.mutationRate.$modelValue,
				  numIterations: this.clusteringParams.advancedOptions.numIterations.$modelValue,
				  populationSize: this.clusteringParams.advancedOptions.populationSize.$modelValue,
				  dataFileLocation: this.dataFileLocation
			  }
		  };
		  console.log(paramObject);
		  var mogaCluster = new Mogacluster(paramObject);
	    mogaCluster.$save(function(response) {
		    $scope.responseFromServer = response.msg;
		    $scope.isCollapsed = true;
		    //$location.path('mogacluster/' + response._id);
		  });

		  this.mogaCluster = '';
	  };

	  $scope.find = function() {
		  Mogacluster.query(function(mogaClusters) {
			  $scope.mogaClusters = mogaClusters;
		  });
	  };

	  $scope.findOne = function() {
		  Mogacluster.get({
			  mogaclusterId: $stateParams.mogaclusterId
		  }, function(mogacluster) {
			  $scope.mogacluster = mogacluster;
			  $scope.paretoImageSrc = "files/public/experiments/" +  mogacluster._id + "/ " + mogacluster.results.pareto_plot_file_name;

		  });
	  };
	  $scope.uploadFileCallback = function(file) {
		  $scope.dataFileName = file.name;
		  $scope.dataFileLocation = file.src
	  };
  }
]);

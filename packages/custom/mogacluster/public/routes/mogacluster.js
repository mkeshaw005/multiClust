'use strict';
/*
angular.module('mean.mogacluster').config(['$stateProvider',
  function($stateProvider) {
    $stateProvider.state('mogacluster example page', {
      url: '/mogacluster/example',
      templateUrl: 'mogacluster/views/index.html'
    });
  }
]);
*/

//Setting up mogacluster
angular.module('mean.mogacluster').config(['$stateProvider',
	function($stateProvider) {
		// Check if the user is connected
		var checkLoggedin = function($q, $timeout, $http, $location) {
			// Initialize a new promise
			var deferred = $q.defer();

			// Make an AJAX call to check if the user is logged in
			$http.get('/loggedin').success(function(user) {
				// Authenticated
				if (user !== '0') $timeout(deferred.resolve);

				// Not Authenticated
				else {
					$timeout(deferred.reject);
					$location.url('/login');
				}
			});

			return deferred.promise;
		};

		$stateProvider.state('mogacluster example page', {
			url: '/mogacluster/example',
			templateUrl: 'mogacluster/views/index.html'
		});

		// states for my app
		$stateProvider
			.state('all mogaclusters', {
				url: '/mogacluster',
				templateUrl: 'mogacluster/views/list.html',
				resolve: {
					loggedin: checkLoggedin
				}
			})
			.state('create mogacluster', {
				url: '/mogacluster/create',
				templateUrl: 'mogacluster/views/create.html',
				resolve: {
					loggedin: checkLoggedin
				}
			})
			.state('edit mogacluster', {
				url: '/mogacluster/:mogaclusterId/edit',
				templateUrl: 'mogacluster/views/edit.html',
				resolve: {
					loggedin: checkLoggedin
				}
			})
			.state('mogacluster by id', {
				url: '/mogacluster/:mogaclusterId/view',
				templateUrl: 'mogacluster/views/view.html',
				resolve: {
					loggedin: checkLoggedin
				}
			});
	}
]);

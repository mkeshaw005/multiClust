'use strict';

angular.module('mean.mogacluster').factory('Mogacluster', ['$resource',
	function($resource) {
		return $resource('mogacluster/:mogaclusterId', {
			mogaclusterId: '@_id'
		}, {
			update: {
				method: 'PUT'
			}
		});
	}
]);

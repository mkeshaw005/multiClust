/**
 * Created by mshaw on 10/17/14.
 */
var mongoose = require('mongoose'),
	MogaclusterModel = mongoose.model('MogaclusterModel'),
_ = require('lodash');

exports.all = function(req, res) {
	MogaclusterModel.find().sort('-created').populate('user', 'name username').exec(function(err, mogoclustermodels) {
		if (err) {
			return res.json(500, {
				error: 'Cannot list the articles'
			});
		}
		res.json(mogoclustermodels);

	});
};

exports.show = function(req, res) {
	res.json(req.mogocluster);
}

exports.mogocluster = function(req, res, next, id) {
	MogaclusterModel.load(id, function(err, mogocluster) {
		if (err) return next(err);
		if (!mogocluster) return next(new Error('Failed to load mogocluster ' + id));
		req.mogocluster = mogocluster;
		next();
	});
};
'use strict';

/**
 * Module dependencies.
 */
var mongoose = require('mongoose'),
	Schema = mongoose.Schema;


/**
 * Article Schema
 */
var MogaclusterSchema = new Schema({
	created: {
		type: Date,
		default: Date.now
	},
	title: {
		type: String,
		required: true,
		trim: true
	},
	analysisParameters: {
		type: Schema.Types.Mixed,
		required: false
	},
	status: {
		type: String,
		required: true
	},
	results: {
		type: Schema.Types.Mixed,
		required: false
	},
	user: {
		type: Schema.ObjectId,
		ref: 'User'
	}
});

/**
 * Validations
 */

/**
 * Statics
 */
MogaclusterSchema.statics.load = function(id, cb) {
	this.findOne({
		_id: id
	}).populate('user', 'name username').exec(cb);
};

mongoose.model('MogaclusterModel', MogaclusterSchema);

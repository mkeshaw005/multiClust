'use strict';

var mongoose = require('mongoose'),
	MogaclusterModel = mongoose.model('MogaclusterModel'),
	mogocluster = require('../controllers/mogocluster'),
	_ = require('lodash'),
	fs = require('fs'),
	path = require('path'),
	mv = require('mv'),
	UPLOAD_DIRECTORY = 'files/public/mogaClustering/',
	EXPERIMENT_DIRECTORIES = rootPath + "/files/public/experiments/"
	;

// The Package is past automatically as first parameter
module.exports = function(Mogacluster, app, auth, database) {


	app.route('/mogacluster')
		.get(mogocluster.all);
		//.post(auth.requiresLogin, articles.create);

	app.route('/mogacluster/:mogaclusterId')
		.get(mogocluster.show);
	app.param('mogaclusterId', mogocluster.mogocluster);
	/*
		.put(auth.requiresLogin, hasAuthorization, articles.update)
		.delete(auth.requiresLogin, hasAuthorization, articles.destroy);
	*/
	// Finish with setting up the articleId param
	//app.param('articleId', articles.article);

  app.get('/mogacluster/example/anyone', function(req, res, next) {
    res.send('Anyone can access this');
  });

  app.get('/mogacluster/example/auth', auth.requiresLogin, function(req, res, next) {
    res.send('Only authenticated users can access this');
  });

  app.get('/mogacluster/example/admin', auth.requiresAdmin, function(req, res, next) {
    res.send('Only users with Admin role can access this');
  });

  app.get('/mogacluster/example/render', function(req, res, next) {
    Mogacluster.render('index', {
      package: 'mogacluster'
    }, function(err, html) {
      //Rendering a view from the Package server/views
      res.send(html);
    });
  });

	app.post('/mogacluster', function(req, res, next) {
		var mogacluster = new MogaclusterModel(req.body);
		mogacluster.user = req.user;
		mogacluster.status = 'Running';
		mogacluster.save(function(err) {
			if (err) {
				console.log("err")
				console.log(err)
				return res.json(500, {
					error: 'Cannot save the Cluster Job'
				});
			}
			var experimentDirectory = EXPERIMENT_DIRECTORIES + mogacluster._id;
			mkdirSync(experimentDirectory);

			var uploadedFilePath = UPLOAD_DIRECTORY +  req.body.analysisParameters.dataFileName;
			var experimintFilePath = experimentDirectory + "/" +  req.body.analysisParameters.dataFileName;
			mv(uploadedFilePath, experimintFilePath, function(err) {
				if (err) {
					console.log("error moving file:");
					console.log(err);
				}
			});
			var PythonShell = require('python-shell');
			var options = {
				mode: 'text',
				//pythonPath: 'path/to/python',
				//pythonOptions: ['-u'],
				scriptPath: app.get('analysisScriptsDir') + '/clustering/thesisMOGA/',
				args: [experimentDirectory, req.body.analysisParameters.dataFileName, parseFloat(req.body.analysisParameters.foldChange), parseInt(req.body.analysisParameters.experimentsOverFoldChange), parseInt(req.body.analysisParameters.populationSize), parseInt(req.body.analysisParameters.numIterations), parseFloat(req.body.analysisParameters.mutationRate)]
			};

			PythonShell.run('runner.py', options, function (err, results) {
				if (err) throw err;
				mogacluster.status = 'Finished';
				mogacluster.results = JSON.parse(results[0]);
				mogacluster.save(function(err) {
					if (err) {
						console.log('Cannot update the Cluster Job')
					}
				});
			});

			res.json(mogacluster);
		});
	});

	var mkdirSync = function (path) {
		try {
			fs.mkdirSync(path);
		} catch(e) {
			if ( e.code != 'EEXIST' ) throw e;
		}
	}

};


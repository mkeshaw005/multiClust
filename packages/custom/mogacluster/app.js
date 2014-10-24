'use strict';

/*
 * Defining the Package
 */
var Module = require('meanio').Module;

var Mogacluster = new Module('mogacluster');

/*
 * All MEAN packages require registration
 * Dependency injection is used to define required modules
 */
Mogacluster.register(function(app, auth, database) {

  //We enable routing. By default the Package Object is passed to the routes
  Mogacluster.routes(app, auth, database);

  //We are adding a link to the main menu for all authenticated users
  Mogacluster.menus.add({
    title: 'Clustering',
    link: 'all mogaclusters',
    roles: ['authenticated'],
    menu: 'main'
  });
  
  Mogacluster.aggregateAsset('css', 'mogacluster.css');

  /**
    //Uncomment to use. Requires meanio@0.3.7 or above
    // Save settings with callback
    // Use this for saving data from administration pages
    Mogacluster.settings({
        'someSetting': 'some value'
    }, function(err, settings) {
        //you now have the settings object
    });

    // Another save settings example this time with no callback
    // This writes over the last settings.
    Mogacluster.settings({
        'anotherSettings': 'some value'
    });

    // Get settings. Retrieves latest saved settigns
    Mogacluster.settings(function(err, settings) {
        //you now have the settings object
    });
    */

  return Mogacluster;
});

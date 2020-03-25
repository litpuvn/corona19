var chart = c3.generate({
    bindto: '#chart',
    data: {
        x: 'x',
       xFormat: '%Y-%m-%d', // 'xFormat' can be used as custom format of 'x'
        columns: []

    },
    axis: {
        x: {
            type: 'timeseries',
            tick: {
                format: '%Y-%m-%d'
            }
        }
    }
});



d3.csv("data/new_cases_march_25.csv")
    .then(function(datarows){

        let chart_data = [];
           // let end_date = new Date(2020, 2, 23);
            let timeseries_data_x = ['x'];
            // let formatTimeX = d3.timeFormat("%m/%d/%Y");
            // let date_string;
            // for (let d = new Date(2020, 0, 22); d <= end_date; d.setDate(d.getDate() + 1)) {
            //     date_string = formatTimeX(d);
            //     timeseries_data.push(date_string);
            // }
            //
            // chart_data.push(timeseries_data);

            let formatTime = d3.timeFormat("%-m/%-d/%Y");
            let timeseries_data = ['world'];
            let worldCases;
        datarows.forEach(function(row, index){

            if (index < 22){ // not working with date before Jan 22, 2020
                return;
            }

            worldCases = +row['World'] || 0;
            timeseries_data.push(worldCases);
            timeseries_data_x.push(row['date'])

        });

        chart_data.push(timeseries_data_x);
        chart_data.push(timeseries_data);
        chart.load({
            columns: chart_data
        });

    })
    .catch(function(error){
     // handle error
        console.log("Error");
        console.log(error);
    });
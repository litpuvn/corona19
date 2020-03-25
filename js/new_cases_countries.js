var chart = c3.generate({
    bindto: '#chart',
    data: {
        x: 'x',
       xFormat: '%m/%d/%Y', // 'xFormat' can be used as custom format of 'x'
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
           let end_date = new Date(2020, 3, 23);
            let cases_count;
            let timeseries_data = ['x'];
            let formatTimeX = d3.timeFormat("%m/%d/%Y");
            let date_string;
            for (let d = new Date(2020, 1, 22); d <= end_date; d.setDate(d.getDate() + 1)) {
                date_string = formatTimeX(d);
                timeseries_data.push(date_string);
            }

            chart_data.push(timeseries_data);

            let countries = {};
            let formatTime = d3.timeFormat("%-m/%-d/%Y");

        datarows.forEach(function(row, index){
            let country = row['Country/Region'];
            // if (country != 'China') {
            //     return;
            // }
            if (countries.hasOwnProperty(country) == true) {
                return;
            }
            countries[country] = true;
            // month has index 0
            let end_date = new Date(2020, 2, 23);
            let cases_count;
            let timeseries_data = [country];
            let timestring = formatTime(end_date);
            // let sum_cases = 0;
            for (let d = new Date(2020, 0, 22); d <= end_date; d.setDate(d.getDate() + 1)) {
                timestring = formatTime(d);
                cases_count = +row[timestring] || 0;
                timeseries_data.push(cases_count);
            }

            // console.log(country + ";sum=" + sum_cases);
            // if (sum_cases > 3000){
            //     chart_data.push(timeseries_data);
            // }



        });

        chart.load({
            columns: chart_data
        });

    })
    .catch(function(error){
     // handle error
        console.log("Error");
        console.log(error);
    });
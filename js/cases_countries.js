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



d3.csv("data/time_series_covid19_confirmed_global_complete.csv")
    .then(function(datarows){

        let chart_data = [];
           let end_date = new Date(2020, 2, 23);
            let cases_count;
            let timeseries_data = ['x'];
            let formatTimeX = d3.timeFormat("%m/%d/%Y");
            let date_string;
            for (let d = new Date(2020, 0, 22); d <= end_date; d.setDate(d.getDate() + 1)) {
                date_string = formatTimeX(d);
                timeseries_data.push(date_string);
            }

            chart_data.push(timeseries_data);

            let countries = {};
            let formatTime = d3.timeFormat("%-m/%-d/%Y");

            // let colorArray = [d3.scale.category20()];
            let colores_g = ["#3366cc", "#dc3912", "#ff9900", "#109618", "#990099", "#0099c6", "#dd4477", "#66aa00", "#b82e2e", "#316395", "#994499", "#22aa99", "#aaaa11", "#6633cc", "#e67300", "#8b0707", "#651067", "#329262", "#5574a6", "#3b3eac"];
            let colorIndex = 0;
            let colors = {};
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
            if (cases_count > 5000){
                chart_data.push(timeseries_data);
                colors[country] = colores_g[colorIndex];
                colorIndex = colorIndex + 1
            }



        });

        chart.load({
            columns: chart_data,
            colors: colors
        });

    })
    .catch(function(error){
     // handle error
        console.log("Error");
        console.log(error);
    });
// Plotly Bar chart
// Ref: https://plotly.com/javascript/reference/bar/
function BarChart(group_data, sub_group_data, cluster, cluster_docs) {
    const width = 600;
    const height = 700;
    const d3colors = d3.schemeCategory10;
    group_data.sort((a, b) => a['Group'] - b['Group']);
    console.log(group_data);    // Three main groups of key phrases
    console.log(sub_group_data);    // Each main group contain a number of sub_groups
    const min_group_id = group_data.reduce((pre, cur) => pre['Group'] < cur['Group'] ? pre : cur)['Group'];
    let thread = 2;
    if (min_group_id === 0) {
        thread = 1;
    }
    const data = create_graph_data();
    console.log(data);

    // Collect top 5 frequent from key phrases
    function collect_title_words(key_phrases, group_id){
        // Create word frequency list
        const create_word_freq_list = function(key_phrases){
            // Create bi-grams
            const create_bi_grams = function(words){
                const bi_grams = [];
                if(words.length === 2){
                    bi_grams.push(words[0] + " " + words[1]);
                }else if(words.length === 3) {
                    bi_grams.push(words[1] + " " + words[2]);
                }
                return bi_grams;
            }
            let word_freq_list = [];
            // Collect word frequencies from the list of key phrases.
            for(const key_phrase of key_phrases){
                const words = key_phrase.split(" ");
                const bi_grams = create_bi_grams(words);
                const n_grams = words.concat(bi_grams);
                // collect uni_gram
                for (const n_gram of n_grams){
                    const range = n_gram.split(" ").length;
                    const found_word = word_freq_list.find(w => w['word'].toLowerCase() === n_gram.toLowerCase())
                    if(found_word){
                        found_word['freq'] += 1;
                    }else{
                        if(n_gram === n_gram.toUpperCase()){
                            word_freq_list.push({'word': n_gram, 'freq':1, 'range': range});
                        }else{
                            word_freq_list.push({'word': n_gram.toLowerCase(), 'freq':1, 'range': range});
                        }
                    }
                }
            }

            return word_freq_list;
            // console.log(word_freq_list);
        }

        const word_freq_list = create_word_freq_list(key_phrases);

        // Update top word frequencies and pick up top words
        const pick_top_words= function(top_words, candidate_words, top_n){
            // Get all the words
            let all_words = top_words.concat(candidate_words);
            // Go through top_words and check if any top word has two words (such as 'twitter data'
            // If so, reduce the frequency of its single word (freq('twitter') -1 and freq('data') -1)
            for(const top_word of top_words){
                if(top_word['range'] === 2){
                    // Get the single word from a bi-gram
                   const top_word_list = top_word['word'].split(" ");
                   // Go through each individual word of top_word
                   for(const individual_word of top_word_list){
                       // Update the frequencies of the single word
                       for(const word of all_words){
                            if(word['word'].toLowerCase() === individual_word.toLowerCase()){
                                // Update the freq
                                word['freq'] = word['freq'] - top_word['freq'];
                            }
                       }
                   }
                }
            }
            // Remove the words of no frequencies
            all_words = all_words.filter(w => w['freq'] > 0);
            // Sort all the words by frequencies
            all_words.sort((a, b) => {
                if(b['freq'] === a['freq']){
                    return b['range'] - a['range'];     // Prefer longer phrases
                }
                return b['freq'] - a['freq'];
            });
            // console.log("All words", all_words.map(w => w['word'] + '(' + w['freq'] + ')'));
            // Pick up top frequent word from all_words
            const new_top_words = all_words.slice(0, top_n);
            const new_candidate_words = all_words.slice(top_n);
            return [new_top_words, new_candidate_words];
        }

        // Pick up top 5 frequent words
        const top_n = 6;
        // Sort by freq
        word_freq_list.sort((a, b) => {
            if(b['freq'] === a['freq']){
                return b['range'] - a['range'];     // Prefer longer phrases
            }
            return b['freq'] - a['freq'];
        });
        // Select top 5 bi_grams as default top_words
        let top_words = word_freq_list.slice(0, top_n);
        let candidate_words = word_freq_list.slice(top_n);
        let is_same = false;
        for(let iteration=0; !is_same && iteration<10; iteration++){
            // console.log("Group id: ", group_id);
            // console.log("Iteration: ", iteration);
            // console.log("top_words:", top_words.map(w => w['word'] + '(' + w['freq'] + ')'));
            const [new_top_words, new_candidate_words] = pick_top_words(top_words, candidate_words, top_n);
            // console.log("new top words: ", new_top_words.map(w => w['word'] + '(' + w['freq'] + ')'));
            // Check if new and old top words are the same
            is_same = true;
            for(const new_word of new_top_words){
                const found = top_words.find(w => w['word'] === new_word['word']);
                if(!found){
                    is_same = is_same && false;
                }
            }
            top_words = new_top_words;
            candidate_words = new_candidate_words;
        }

        // Return the top 3
        return top_words.slice(0, 5).map(w => w['word']);
    }


    // Graph data
    function create_graph_data() {
        let data = [];
        // Re-order the groups to match with the order of the chart.
        for (let i =0; i< group_data.length; i++) {
            const group = group_data[i];
            const group_id = group['Group'];
            const group_name = "Group#" + (group_id + thread);
            // Get the sub-group
            const sub_groups = sub_group_data.filter(g => g['Group'] === group_id);
            // create a trace
            let trace = {
                x: [], y: [], text: [],
                orientation: 'h', type: 'bar',
                name: group_name,
                textposition: 'inside',
                insidetextanchor: "start",
                insidetextfont: {
                    size: 14
                },
                outsidetextfont:{
                    size: 14
                },
                hovertemplate: '%{text} (%{x} papers)',
                marker: {
                    color: d3colors[group_id + 1]
                }
            };
            if(i > 0){
                trace['xaixs'] = "x" + (i+1)
                trace['yaxis'] = "y" + (i+1)
            }else{
                trace['xaixs'] = "x"
                trace['yaxis'] = "y"
            }

            if (sub_groups.length > 0) {
                for (const sub_group of sub_groups) {
                    // console.log(sub_group);
                    const sub_group_id = sub_group['SubGroup'];
                    // const title_words = sub_group['TitleWords'];
                    const key_phrases = sub_group['Key-phrases'];
                    // Get the title words of a sub-group
                    const title_words = collect_title_words(key_phrases, sub_group_id);
                    sub_group['TitleWords'] = title_words;
                    // Update the title word of a sub-group;
                    const num_docs = sub_group['NumDocs'];
                    trace['y'].push(group_name + "|" + sub_group_id);
                    trace['x'].push(num_docs);
                    trace['text'].push('<b>' + title_words.slice(0,3).join(", ") + '</b>');
                }
            } else {
                // Add the group
                // const title_words = group['TitleWords'];
                const title_words = collect_title_words(group['Key-phrases'], group_id);
                group['TitleWords'] = title_words;
                const num_docs = group['NumDocs'];
                trace['y'].push(group_name + "#" + group_id);
                trace['x'].push(num_docs);
                trace['text'].push('<b>' + title_words.slice(0,3).join(", ") + '</b>')
            }
            data.push(trace);
        }

        return data;
    }

    // Populate the bar height
    function populate_y_axis_domain(layout){
        let total_sub_groups = 0;
        // Go through each main group
        for (let i =0; i< group_data.length; i++) {
            const group = group_data[i];
            const group_id = group['Group'];
            // Get the sub-group
            const sub_groups = sub_group_data.filter(g => g['Group'] === group_id);
            if(sub_groups.length > 0){
                total_sub_groups += sub_groups.length;
            }else{
                // The group does not have a sub-group
                total_sub_groups += 1;
            }
        }
        // Get the portion of each sub-group
        // let portion = Math.min(1.0 / total_sub_groups * 0.85, 0.2);
        const portion = 0.08;
        const gap = 0.05;
        console.log("portion", portion);
        // let gap = 0.02;
        // if(group_data.length > 1){
        //     gap = (1.0 - (portion * total_sub_groups))/(group_data.length-1);    // Gap between different groups
        // }
        // console.log("gap", gap);
        let cur_domain = 1.0;
        for(let i=group_data.length-1; i >=0; i--){
            const group = group_data[i];
            const group_id = group['Group'];
            // Get the sub-group
            const sub_groups = sub_group_data.filter(g => g['Group'] === group_id);
            // Add one base portion
            let next_domain = cur_domain - portion;
            if(sub_groups.length > 0){
                // The domain Proportion to the number of sub-groups
                next_domain = cur_domain - (sub_groups.length * portion);
            }
            // Get axis name
            const axis_name =  (i > 0 ? "yaxis" + (i+1) : "yaxis");
            layout[axis_name] ={
                tickfont: {
                    size: 1,
                },
                domain: [next_domain, cur_domain]
            }
            cur_domain = next_domain - gap;// Add the gap to separate different groups
        }
        console.log(layout);
        return layout;
    }

    function create_UI() {
        let layout = {
            xaxis: {
                title: 'Number of Papers',
                domain: [0, 1.0],
            },
            width: width,
            height: height,
            autosize: true,
            showlegend: false,
            margin: {"l": 0, "r": 0, "t": 0},
            legend: { traceorder: 'reversed'},
            grid: {
                rows: (group_data.length >=3 ? group_data.length: 3),   // Display three rows by default
                columns: 1,
                pattern: 'independent',
                roworder: 'top to bottom'
            },

        };
        console.log(data);
        layout = populate_y_axis_domain(layout);
        console.log(layout);
        const config = {
            displayModeBar: false // Hide the floating bar
        }
        // Plot bar chart
        Plotly.newPlot('key_phrase_chart', data, layout, config);
        const chart_element = document.getElementById('key_phrase_chart');
        // // Define the hover event
        chart_element.on('plotly_click', function (data) {
            $('#sub_group').empty();
            $('#doc_list').empty();
            const id = data.points[0].y;
            // Get the marker
            const marker = data.points[0].data.marker;
            const color = marker.color;
            console.log(id);
            if (id.includes("#")) {
                const group_id = parseInt(id.split("#")[1]) - thread;
                // Get the sub-group
                if (id.includes("|")) {
                    const subgroup_id = parseInt(id.split("|")[1]);
                    const sub_group = sub_group_data.find(g => g['Group'] === group_id && g['SubGroup'] === subgroup_id);
                    if (sub_group) {
                        const view = new KeyPhraseView(sub_group, cluster_docs, color);
                    }
                } else {
                    const found = id.match(/#/g);
                    if (found && found.length === 2) {
                        // This indicates the groups has only one subgroup. so we use the group data.
                        // Get the group
                        const group = group_data.find(g => g['Group'] === group_id);
                        // Display the group
                        const view = new KeyPhraseView(group, cluster_docs, color);
                    }
                }
            }
        });// End of chart onclick event

    }

    create_UI();
}
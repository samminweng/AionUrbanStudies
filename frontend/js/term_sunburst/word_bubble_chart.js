// Create a word network chart using highchart library
// Ref: https://www.highcharts.com/docs/chart-and-series-types/network-graph
// API: https://api.highcharts.com/highcharts/
function WordBubbleChart(group, cluster_docs, color) {
    const d3colors = d3.schemeCategory10;
    const title_words = group['TitleWords'];
    const group_key_phrases = group['Key-phrases'];
    const group_docs = cluster_docs.filter(d => group['DocIds'].includes(d['DocId']));
    const word_key_phrase_dict = collect_key_phrases_by_words();
    const word_neighbour_phrases_dict = collect_word_neighbour_phrases();
    const phrase_doc_dict = collect_phrase_docs();
    console.log("group", group);
    console.log("group docs", group_docs);
    console.log("word_key_phrase_dict", word_key_phrase_dict);
    console.log("word_neighbour_phrases_dict", word_neighbour_phrases_dict);
    console.log("phrase_doc_dict", phrase_doc_dict);

    // Collect the phrase to doc relation
    function collect_phrase_docs(){
        let dict = {};
        for(const doc of group_docs){
            const doc_id = doc['DocId'];
            const key_phrases = doc['KeyPhrases'];
            for(const key_phrase of key_phrases){
                if(key_phrase.toLowerCase() in dict){
                    dict[key_phrase.toLowerCase()].push(doc_id);
                }else{
                    dict[key_phrase.toLowerCase()] = [doc_id];
                }
            }
        }
        return dict;
    }
    // Collect all key phrases containing title words
    function collect_key_phrases_by_words() {
        let dict = {}; // Key: word, Value: a list of key phrases
        for (const word of title_words) {
            dict[word] = [];
            for (const key_phrase of group_key_phrases) {
                if (key_phrase.toLowerCase().includes(word.toLowerCase())) {
                    dict[word].push(key_phrase);
                }
            }
        }
        return dict;
    }

    // Collect all key phrases co-occurring with title words
    function collect_word_neighbour_phrases() {
        // Collect the relation between key phrases and its neighbour key phrases
        const collect_other_key_phrases_by_phrase = function () {
            let dict = {};
            for (const key_phrase of group_key_phrases) {
                // Get the docs containing key phrase
                // Get the neighbour phrases
                let neighbour_phrases = [];
                for (const doc of group_docs) {
                    const doc_key_phrases = doc['KeyPhrases'];
                    const found = doc_key_phrases.find(kp => kp.toLowerCase() === key_phrase.toLowerCase());
                    if (found) {
                        const other_phrases = doc_key_phrases.filter(kp => kp.toLowerCase() !== key_phrase.toLowerCase());
                        neighbour_phrases = neighbour_phrases.concat(other_phrases);
                    }
                }
                dict[key_phrase.toLowerCase()] = neighbour_phrases;
            }
            return dict;
        };

        // Extract the common words from a list of key phrases
        const extract_words_from_phrases = function (phrases) {
            let neighboring_words = [];
            for (const phrase of phrases) {
                const words = phrase.toLowerCase().split(" ");
                const last_word = words[words.length - 1];
                const found = neighboring_words.find(w => w['word'] === last_word);
                if (!found) {
                    neighboring_words.push({'word': last_word, 'phrases': [phrase]})
                } else {
                    found['phrases'].push(phrase);
                }
            }
            // Sort by the number of phrases
            neighboring_words.sort((a, b) => b['phrases'].length - a['phrases'].length);
            return neighboring_words;
        };

        // A dict stores the relation between a key phrases and its neighbouring phrases
        const other_phrases_dict = collect_other_key_phrases_by_phrase();
        let dict = {};
        for (const title_word of title_words) {
            // Keep top 10 neighbouring words

            // Get the key phrases about title word
            const key_phrases = word_key_phrase_dict[title_word];
            let neighboring_phrases = [];
            // Aggregate all neighbouring key phrases of key phrases
            for (const key_phrase of key_phrases) {
                // Get the neighbouring key phrases
                const other_phrases = other_phrases_dict[key_phrase.toLowerCase()].map(kp => kp.toLowerCase());
                neighboring_phrases = neighboring_phrases.concat(other_phrases.map(p => p.toLowerCase()));

                // dict[title_word].push({'phrase': key_phrase, 'neighboring_words': neighboring_words});
            }
            // Remove duplicated
            neighboring_phrases = [...new Set(neighboring_phrases)];
            // Obtain the common words from a list of neighbouring phrases
            let neighbouring_words = extract_words_from_phrases(neighboring_phrases);
            // Removed the duplicated title words, such as 'data'
            // neighbouring_words = neighbouring_words.filter(w => ! title_words.includes(w['word']));
            // Get top 5 neighbouring words
            dict[title_word] = neighbouring_words.slice(0, 5);
        }
        return dict;
    }

    // Get all the nodes
    function collect_nodes_links(){
        let nodes = [];
        let links = [];
        for(let i=0; i< title_words.length; i++){
            const title_word = title_words[i];
            const key_phrases = word_key_phrase_dict[title_word];
            nodes.push({
                id: title_word,
                color: color,
                marker: {
                    radius: Math.min(key_phrases.length + 10, 50)
                },
                dataLabels: {
                    backgroundColor: color,
                    allowOverlap: false,
                    style: {
                        color: 'white',
                        fontSize: '16px',
                        textOutline: true
                    }
                }
            });

            const neighbor_words = word_neighbour_phrases_dict[title_word];
            for(const n_word of neighbor_words){
                const n_phrases = n_word['phrases'];
                // Add an intermediate node
                let parent = title_word;
                if(n_phrases.length > 1){
                    // Add the node
                    nodes.push({
                        id: n_word['word'],
                        marker: {
                            radius: n_phrases.length+ 5
                        },
                        color: 'gray',
                        dataLabels: {
                            style: {
                                fontSize: '10px'
                            }
                        }
                    });
                    // Add link
                    links.push({
                        from: title_word, to: n_word['word']
                    });
                    parent = n_word['word'];
                }


                // Add phrases
                for(const phrase of n_phrases){
                    // found if the phrase exists
                    const found = nodes.find(n => n['id'] === phrase);
                    if(!found){
                        nodes.push({
                            id: phrase,
                            marker: {
                                radius: n_phrases.length + 5
                            },
                            color: 'gray',
                            dataLabels: {
                                style: {
                                    fontSize: '9px'
                                }
                            }
                        });
                    }
                    // Add link
                    links.push({
                        from: parent, to: phrase
                    });
                }
            }
        }

        return [nodes, links];

    }


    function _createUI() {
        $('#term_occ_chart').empty();
        const [nodes, links] = collect_nodes_links();

        Highcharts.chart('term_occ_chart', {
            chart: {
                type: 'networkgraph',
                height: 600,
            },
            title: {
                text: ''
            },
            events:{
                click: function(event){
                    console.log(event);
                }
            },
            plotOptions: {
                networkgraph: {
                    keys: ['from', 'to'],
                    layoutAlgorithm: {
                        enableSimulation: false,
                        // linkLength: 10,
                        // friction: -0.1,
                        approximation: 'barnes-hut',
                    }
                }
            },
            series: [{
                dataLabels: {
                    enabled: true,
                    linkFormat: '',
                    allowOverlap: false,
                    style: {
                        color:"black",
                        textOutline: false
                    }
                },
                data: links,
                nodes: nodes
            }]
        });
        //


    }

    _createUI();

}

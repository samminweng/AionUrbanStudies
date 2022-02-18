// Create a word network chart using highchart library
// Ref: https://www.highcharts.com/docs/chart-and-series-types/network-graph
// API: https://api.highcharts.com/highcharts/
function WordBubbleChart(group, cluster_docs, color) {
    const d3colors = d3.schemeCategory10;
    const title_words = group['TitleWords'];
    const group_key_phrases = group['Key-phrases'];
    const group_docs = cluster_docs.filter(d => group['DocIds'].includes(d['DocId']));
    const word_key_phrase_dict = Utility.create_word_key_phrases_dict(title_words, group_key_phrases);
    const word_doc_dict = Utility.create_word_doc_dict(title_words, group_docs, word_key_phrase_dict);
    const phrase_doc_dict = Utility.collect_phrase_docs(group_docs);
    console.log("group", group);
    console.log("group docs", group_docs);
    console.log("word_key_phrase_dict", word_key_phrase_dict);
    console.log("word_doc_dict", word_doc_dict);
    console.log("phrase_doc_dict", phrase_doc_dict);


    // Get all the nodes
    function collect_nodes_links() {
        let nodes = [];
        let links = [];
        for (let i = 0; i < title_words.length; i++) {
            const title_word = title_words[i];
            // Check if the word appear in word_neighbour_phrases_dict

            // const key_phrases = word_key_phrase_dict[title_word];
            nodes.push({
                id: title_word,
                color: color,
                marker: {
                    radius: 50
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
            events: {
                click: function (event) {
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
                        integration: 'euler',

                    }
                }
            },
            series: [{
                dataLabels: {
                    enabled: true,
                    linkFormat: '',
                    allowOverlap: false,
                    style: {
                        color: "black",
                        textOutline: false,
                        style: {
                            fontSize: '9px'
                        }
                    }
                },
                data: links,
                nodes: nodes
            }]
        });
    }

    _createUI();

}




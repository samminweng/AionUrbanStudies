// Create a word network chart using highchart library
// Ref: https://www.highcharts.com/docs/chart-and-series-types/network-graph
// API: https://api.highcharts.com/highcharts/
function WordBubbleChart(group, cluster_docs, color) {
    const d3colors = d3.schemeCategory10;
    const title_words = group['TitleWords'].concat(['others']);
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

    // Highlight key terms
    function mark_key_terms(div, terms, class_name) {
        if(terms !== null){
            // Check if the topic is not empty
            for (const term of terms) {
                // Mark the topic
                const mark_options = {
                    "separateWordSearch": false,
                    "accuracy": {
                        "value": "partially",
                        "limiters": [",", ".", "'s", "/", ";", ":", '(', ')', '‘', '’', '%', 's', 'es']
                    },
                    "acrossElements": true,
                    "ignorePunctuation": ":;.,-–—‒_(){}[]!'\"+=".split(""),
                    "className": class_name
                }
                div.mark(term, mark_options);
            }
        }
        return div;
    }

    // Create a tooltip
    function create_tooltip(id, text){
        // console.log("this.point", this.point);
        let div = $('<div>' + text + '</div>');
        if(!id.includes("paper#")){
            // Highlight title word
            const title_word = id;
            div = mark_key_terms(div, [title_word], 'key_term');
        }else{
            const doc_id = parseInt(id.split("paper#")[1]);
            // Get the title words linking to this doc
            let words = [];
            for(const title_word of title_words){
                if(word_doc_dict[title_word].includes(doc_id)){
                    words.push(title_word);
                }
            }
            div = mark_key_terms(div, words, 'key_term');
        }
        return div.html();
    }

    // Display a detail chart for title word
    function display_word_paper_chart(title_word){
        console.log(title_word);
        // Get all the nodes
        const collect_nodes_links = function(word_docs) {
            let nodes = [];
            let links = [];
            // Create a title word node
            const key_phrases = word_key_phrase_dict[title_word];
            nodes.push({
                id: title_word,
                name: title_word + "<br>(" + word_docs.length + " papers)",
                text: title_word + "<br>(" + word_docs.length + " papers)",
                color: color,
                marker: {
                    radius: Math.min(Math.sqrt(key_phrases.length) * 5 + 10, 30)
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
            // Add word node and link between word and paper
            // Add link
            for(let i=0; i< word_docs.length;i++) {
                const doc = word_docs[i];
                const doc_id = doc['DocId'];
                const word_key_phrase = doc['KeyPhrases'].find(phrase => key_phrases.includes(phrase));
                // const key_phrases = doc['KeyPhrases'].filter(phrase => phrase !== word_key_phrase);
                const text = doc['KeyPhrases'].join(";<br>");
                const id = "paper#" + doc_id
                nodes.push({
                    id: id,
                    name: "<span>" + word_key_phrase + "</span>",
                    text: text,
                    color: 'gray',
                    // opacity: 0.5,
                    marker: {
                        radius: 20
                    },
                    dataLabels: {
                        format: '{point.name}',
                        style: {
                            color: 'black',
                            fontSize: '12px',
                            textOutline: true,
                        },
                        align: 'left',
                        useHTML: true,
                        verticalAlign: 'bottom',
                    }
                });
                const link = {from: title_word, to: id}
                links.push(link);
            }
            return [nodes, links];
        };

        // $('term_occ_chart').empty();
        const word_docs = group_docs.filter(d => word_doc_dict[title_word].includes(d['DocId']));
        const [nodes, links] = collect_nodes_links(word_docs);
        $('#term_occ_chart').empty();
        Highcharts.chart('term_occ_chart', {
            chart: {
                type: 'networkgraph',
                height: 600,
            },
            title: {
                text: ''
            },
            tooltip:{
                backgroundColor: 'white',
                borderColor: 'black',
                borderRadius: 10,
                borderWidth: 3,
                formatter: function() {
                    return create_tooltip(this.point.id, this.point.text);
                }
            },
            plotOptions: {
                networkgraph: {
                    keys: ['from', 'to'],
                    layoutAlgorithm: {
                        enableSimulation: false,
                        linkLength: 150,
                        integration: 'verlet',
                        approximation: 'barnes-hut',
                        // gravitationalConstant: 0.1
                    }
                }
            },
            series: [{
                dataLabels: {
                    enabled: true,
                    linkFormat: '',
                    allowOverlap: false
                },
                data: links,
                nodes: nodes
            }]
        });
    }

    // Display the papers for all words
    function display_all_word_chart(){
        $('#term_occ_chart').empty();
        // Get all the nodes
        const collect_nodes_links = function() {
            let nodes = [];
            let links = [];
            // // Add the paper
            for(let i=0; i< group_docs.length;i++){
                const doc = group_docs[i];
                const doc_id = doc['DocId'];
                const text = doc['KeyPhrases'].join(";<br>");
                nodes.push({
                    id: "paper#" + doc_id,
                    name: '',
                    text: text,
                    color: 'gray',
                    marker: {
                        radius: 5
                    },
                    dataLabels: {
                        enabled: false,
                    }
                });
            }
            // Add word node and link between word and paper
            for (let i = 0; i < title_words.length; i++) {
                const title_word = title_words[i];
                const word_docs = word_doc_dict[title_word];
                // Check if the word appear in word_neighbour_phrases_dict
                const key_phrases = word_key_phrase_dict[title_word];
                nodes.push({
                    id: title_word,
                    name: "<div>" + title_word + ' (' + word_docs.length + ')' + "</div>",
                    text: "<div>" + key_phrases.join(";<br>") + '</div>',
                    color: color,
                    marker: {
                        radius: Math.min(Math.sqrt(key_phrases.length) * 5 + 10, 30)
                    },
                    dataLabels: {
                        backgroundColor: color,
                        allowOverlap: false,
                        style: {
                            color: 'white',
                            fontSize: '14px',
                            textOutline: true
                        }
                    }
                });
                // Add link
                for(const doc_id of word_docs){
                    const link ={from: title_word, to: "paper#" + doc_id}
                    links.push(link);
                }

            }

            return [nodes, links];

        };
        const [nodes, links] = collect_nodes_links();
        // Create highcharts
        Highcharts.chart('term_occ_chart', {
            chart: {
                type: 'networkgraph',
                height: 600,
            },
            title: {
                text: ''
            },
            tooltip:{
                backgroundColor: 'white',
                borderColor: 'black',
                borderRadius: 10,
                borderWidth: 3,
                opacity: 0,
                formatter: function() {
                    return create_tooltip(this.point.id, this.point.text);
                }
            },
            plotOptions: {
                networkgraph: {
                    keys: ['from', 'to'],
                    layoutAlgorithm: {
                        enableSimulation: false,
                        linkLength: 25,
                        // friction: -0.1,
                        approximation: 'barnes-hut',
                        integration: 'euler',
                    }
                }
            },
            series: [{
                dataLabels: {
                    enabled: true,
                    // align: 'center',
                    linkFormat: '',
                    allowOverlap: false,
                    style: {
                        color: "black",
                        textOutline: false,
                        textAlign: 'center',
                        style: {
                            fontSize: '9px'
                        },
                        useHTML: true,
                    }
                },
                data: links,
                nodes: nodes
            }]
        });

        // Add onclick event
        document.getElementById('term_occ_chart').addEventListener('click', e => {
            // console.log(e);
            const id = e.point.id;
            // Check if clicking on any title word
            const found = title_words.find(w => w === id);
            if(found){
                display_word_paper_chart(found);
            }
        });
    }

    function _createUI() {
        // display_word_paper_chart("others");
        display_all_word_chart();
        // Add click event
        $('#back_btn').button();
        $('#back_btn').click(function(event){
            display_all_word_chart();
        });
    }

    _createUI();

}




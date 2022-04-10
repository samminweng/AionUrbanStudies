// Create a word network chart using Open sourced Vis.js
// Ref: https://almende.github.io/vis/index.html
// API: https://visjs.org/
function WordBubbleChart(group, cluster_docs, color) {
    const height = 400;
    const topic_words = group['TopicWords'].concat(["others"]);
    const group_key_phrases = group['Key-phrases'];
    const group_docs = cluster_docs.filter(d => group['DocIds'].includes(d['DocId']));
    const word_key_phrase_dict = Utility.create_word_key_phrases_dict(topic_words, group_key_phrases);
    const word_doc_dict = Utility.create_word_doc_dict(topic_words, group_docs, word_key_phrase_dict);

    // Highlight key terms
    function mark_key_terms(div, terms, class_name) {
        if (terms !== null) {
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

    // Display a detail chart for title word
    function display_word_paper_chart(topic_word) {
        $('#term_occ_chart').empty();
        // console.log(topic_word);
        // Get all the nodes
        const collect_nodes_links = function (word_docs) {
            let nodes = [];
            let links = [];
            // Create a title word node
            const key_phrases = word_key_phrase_dict[topic_word];
            nodes.push({
                id: topic_word,
                label: topic_word,
                color: color,
                font:{
                    color: 'white',
                    size: 20
                }
            });
            // Add word node and link between word and paper
            // Add link
            for (let i = 0; i < word_docs.length; i++) {
                const doc = word_docs[i];
                const doc_id = doc['DocId'];
                const key_phrase = doc['KeyPhrases'].find(phrase => key_phrases.includes(phrase));
                let title_div = $("<div>" + doc['KeyPhrases'].join("<br>") + "</div>");
                // Mark topic word
                title_div = mark_key_terms(title_div, [topic_word], 'key_term');
                const id = "paper#" + doc_id
                nodes.push({
                    id: id,
                    label: key_phrase,
                    // shape: 'icon',
                    shape: 'box',
                    color: 'gray',
                    font:{
                        color: 'white',
                        size: 14,
                        multi: true
                    },
                    // icon:{
                    //     face: "'FontAwesome'",
                    //     code: '\uf15c',
                    //     color: 'gray',
                    // },
                    title: title_div.html()
                });
                const link = {from: topic_word, to: id}
                links.push(link);
            }
            return [nodes, links];
        };

        // $('term_occ_chart').empty();
        const word_docs = group_docs.filter(d => word_doc_dict[topic_word].includes(d['DocId']));
        const [nodes, links] = collect_nodes_links(word_docs);
        const chart_container = document.getElementById('term_occ_chart');
        const data = {
            nodes: nodes,
            edges: links
        };
        const options = {
            autoResize: true,
            height: '100%',
            width: '100%'
        };
        // Create the network graph
        const network = new vis.Network(chart_container, data, options);

        // Add button
        const btn = $('<button type="button" class="btn btn-link">Back to Previous</button>');
        btn.button();
        btn.on('click', function(){
            display_all_word_bubble_chart();
        });
        $('#button_area').append(btn);
        const select_index = topic_words.indexOf(topic_word);
        // Display the key phrase view
        const view = new KeyPhraseView(group, cluster_docs, select_index);

    }

    // Display the papers for all words
    function display_all_word_bubble_chart() {
        $('#button_area').empty();
        $('#term_occ_chart').empty();
        // Get all the nodes
        const collect_nodes_edges = function () {
            let nodes = [];
            let edges = [];
            // // Add the paper
            for (let i = 0; i < group_docs.length; i++) {
                const doc = group_docs[i];
                const doc_id = doc['DocId'];
                // console.log(doc['KeyPhrases']);
                nodes.push({
                    id: "article#" + doc_id,
                    label: '',
                    color: 'gray'
                });
            }
            // Collect the mapping between doc_id and topic word
            let doc_word_dict = {};
            // Add word node and link between word and paper
            for (let i = 0; i < topic_words.length; i++) {
                const topic_word = topic_words[i];
                const word_docs = word_doc_dict[topic_word];
                // Check if the word appear in word_neighbour_phrases_dict
                const key_phrases = word_key_phrase_dict[topic_word];
                if (key_phrases.length > 0) {
                    let title_div = $("<div>" + key_phrases.join("<br>") + "</div>");
                    // Mark topic word
                    title_div = mark_key_terms(title_div, [topic_word], 'key_term');
                    // Determine the font sizes based on the number of key phrases
                    const font_size = Math.min(Math.sqrt(key_phrases.length) * 5 + 15, 30);
                    nodes.push({
                        id: "word#" + topic_word,
                        label: topic_word,
                        color: color,
                        font: {
                            color: 'white',
                            size: font_size
                        },
                        title: title_div.html(),
                    });
                    // Add link
                    for (const doc_id of word_docs) {
                        const link = {from: "word#" + topic_word, to: "article#" + doc_id}
                        edges.push(link);
                        if (!(doc_id in doc_word_dict)) {
                            doc_word_dict[doc_id] = [];
                        }
                        // Add the word to doc_id list
                        if (!doc_word_dict[doc_id].includes(topic_word)) {
                            doc_word_dict[doc_id].push(topic_word)
                        }
                    }
                }
            }
            // console.log(doc_word_dict);
            // Add title to each doc node
            for (const [doc_id, words] of Object.entries(doc_word_dict)) {
                try {
                    // Get the doc node
                    let doc_node = nodes.find(n => n['id'] === 'article#' + doc_id);
                    const doc = group_docs.find(d => d['DocId'] === parseInt(doc_id));
                    let title_div = $("<div>" + doc['KeyPhrases'].join("<br>") + "</div>");
                    // Mark topic word
                    title_div = mark_key_terms(title_div, [words], 'key_term');
                    doc_node['title'] = title_div.html();
                } catch (error) {
                    console.error(error);
                }
            }
            // console.log(nodes);

            return [nodes, edges];

        };
        const [nodes, links] = collect_nodes_edges();
        const chart_container = document.getElementById('term_occ_chart');
        // Set the node and edges
        const data = {
            nodes: nodes,
            edges: links
        };
        const options = {
            autoResize: true,
            height: '100%',
            width: '100%'
        };
        // Create the network graph
        const network = new vis.Network(chart_container, data, options);
        //  Onclick event
        network.on('click', function (properties) {
            const node = properties.nodes[0];
            // Check if node is a topic word
            if (node && node.startsWith('word#')) {
                const topic_word = node.split("#")[1];
                display_word_paper_chart(topic_word);
                // console.log(topic_word);
            }
        });
        $('#back_btn').hide();
    }

    function _createUI() {
        // display_word_paper_chart("others");
        display_all_word_bubble_chart();
    }

    _createUI();

}




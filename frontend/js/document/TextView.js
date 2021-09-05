'use restrict';

// Create a story div and highlight the entities
function _TextView(_story) {
    const story = _story;
    let container = $('<div> "' + story['story'] + '"</div>');
    this.getContainer = function () {
        return container;
    };

    this.highlightEntities = function () {
        // Highlight each entity
        for (let entity of entities) {
            let text = entity['text'];
            let ner = entity['ner'];
            let wikiLink = entity['wiki'];
            container.mark(text, {
                "element": "a",
                "className": "entity",
                "separateWordSearch": false,
                "accuracy": {
                    "value": "exactly",
                    "limiters": [",", ".", "'s"]
                },
                "caseSensitive": true,
                // "ignorePunctuation":puncts,
                "each": function (node) {
                    // Add
                    // let link = $('<a target="_blank" href="https://en.wikipedia.org/wiki/'+wikiLink+'">'+wikiLink+'</a>');
                    // let link = document.createElement('a');
                    node.setAttribute("target", "_blank");
                    node.setAttribute("href", "https://en.wikipedia.org/wiki/" + text.replace(" ", "_"));
                    // node.appendChild(link);
                    // let title = ;
                    // // Add tooltip to each element, e.g. data-toggle="tooltip" data-placement="top" title="Tooltip on top"
                    node.setAttribute("data-toggle", "tooltip");
                    node.setAttribute("data-html", "true");
                    node.setAttribute("title", "'" + text + "' is detected as " + ner + "");
                },
            });

        }
    }


}

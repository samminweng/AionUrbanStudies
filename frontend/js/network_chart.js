function NetworkChart(_corpus_data, _collocation_data, _occurrence_data, _starting_year) {
    const starting_year = _starting_year; //
    const corpus_data = _corpus_data;
    const collocation_data = _collocation_data; // Describe the terms
    const occurrence_data = _occurrence_data.find(occ => occ['starting_year'] === starting_year); // Describe the number of document ids between two terms
    console.log(occurrence_data)
    function _createUI() {
         // Create the nodes and links of term occurrences.
        let d3_network_graph = new D3NetworkGraph(collocation_data, occurrence_data, corpus_data);
        // Create a document list view for a collocation
        let doc_list_view = new DocumentListView('machine learning', collocation_data, corpus_data);
    }

    _createUI();
}

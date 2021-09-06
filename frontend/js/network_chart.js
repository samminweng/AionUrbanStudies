function NetworkChart(_corpus_data, _collocation_data, _occurrence_data) {
    const corpus_data = _corpus_data;
    const collocation_data = _collocation_data; // Describe the terms
    const occurrence_data = _occurrence_data; // Describe the number of document ids between two terms

    // Create a document list view for a collocation
    function _create_collocation_document_list_view(node_name) {
        console.log(node_name);
        // For testing, we create a document list view for 'machine learning'
        let collocation = collocation_data.find(({Collocation}) => Collocation === node_name);
        let documents = Utility.collect_documents(collocation, corpus_data);
        // console.log(documents);
        let doc_list_view = new DocumentListView(documents);
    }


    function _createUI() {
         // Create the nodes and links of term occurrences.
        let d3_network_graph = new D3NetworkGraph(collocation_data, occurrence_data);
        _create_collocation_document_list_view('machine learning');

    }

    _createUI();
}

class Utility {
    // Collect all the document ids for a collocation
    static collect_documents(collocation, corpus_data){
        let col_doc_ids = collocation['DocIDs'];
        // Collect all the document ids
        let doc_id_set = new Set();
        for(const year in col_doc_ids){
            let doc_ids = col_doc_ids[year];
            doc_ids.forEach(doc_id => doc_id_set.add(doc_id));
        }
        let documents = corpus_data.filter(doc => doc_id_set.has(doc['DocId']));    // Get the documents 
        return documents;
    }

    // Convert the collocations json to the format of D3 network graph
    static create_node_link_data(collocation_data, occurrence_data){
        let doc_id_set = new Set(); // Store all the document ids
        let max_doc_ides = 0;
        // Populate the nodes with collocation data
        let nodes = [];
        for(let collocation of collocation_data){
            let node = {'id': collocation['index'], 'name': collocation['Collocation']}
            nodes.push(node);
            // Add the doc_ids to doc_id_sets
            let col_doc_ids = collocation['DocIDs'];
            let total_doc_ids = 0;
            for(const year in col_doc_ids){
                const doc_ids = col_doc_ids[year];
                for(const doc_id in doc_ids){
                    doc_id_set.add(doc_id);
                }
                total_doc_ids += doc_ids.length;
            }
            if (max_doc_ides < total_doc_ids){
                max_doc_ides = total_doc_ids;
            }
        }
        // console.log(doc_id_set);
        // Populate the links with occurrences
        const occurrences = occurrence_data['occurrences'];
        let links = [];
        let max_link_length = 0;
        for (let source=0; source < nodes.length; source++) {
            for (let target =0; target < nodes.length; target++){
                if (source !== target){
                    let occ = occurrences[source][target];
                    if(occ.length > 0){
                        // Add the link
                        links.push({source: source, target: target});
                        // Update the max links
                        if(max_link_length < occ.length){
                            max_link_length = occ.length;
                        }
                    }
                }
            }
        }
        return {node_link_data: {nodes: nodes, links: links}, doc_id_set: doc_id_set,
            max_doc_ides: max_doc_ides, max_link_length: max_link_length};
    }
}

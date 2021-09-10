class TermChartUtility {
    // Collect all the document
    static collect_documents_by_doc_ids(collocation, doc_term_data){
        const doc_ids = Object.values(collocation['DocIDs']).flat();
        console.log(doc_ids);
        return doc_term_data.filter(doc_term => doc_ids.includes(doc_term['DocId']));
    }
    // Get the maximal doc length
    static get_max_node_size(nodes){
        let max_node_size = 0;
        for(let node of nodes){
            const doc_ids = node['doc_ids'];
            max_node_size = Math.max(max_node_size, doc_ids.length);
        }
        return max_node_size;
    }
    // Convert the collocations json to the format of D3 network graph
    static create_node_link_data(term_map, occurrences){
        // Populate the nodes with collocation data
        let nodes = [];
        for(let i = 0; i < term_map.length; i++){
            const tm = term_map[i];
            const col_name = tm[0];
            const doc_ids = tm[1];
            const node = {'id': i, 'name': col_name, 'doc_ids': doc_ids};

            nodes.push(node);
        }
        // Populate the links with occurrences
        let links = [];
        for (let source=0; source < nodes.length; source++) {
            for (let target =0; target < nodes.length; target++){
                if(source !== target){
                    const occ = occurrences[source][target];
                    if(occ.length > 0){
                        // Add the link
                        links.push({source: source, target: target, value: occ.length});
                    }
                }
            }
        }
        // Return
        return {nodes: nodes, links: links};
    }
    static group_0 = ['machine learning', 'neural network', 'random forest', 'artificial intelligence', 'deep learning'];
    // Get the group number
    static get_group_number(collocation){
        if (this.group_0.includes(collocation)) {
            return 0;
        }
        return 1;
    }
}




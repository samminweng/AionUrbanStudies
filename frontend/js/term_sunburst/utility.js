// Utility functions
class Utility {
    // Create a dict to store the relation between word and doc
    static create_word_doc_dict(title_words, docs, word_key_phrase_dict) {
        let dict = {};
        for (const [title_word, key_phrases] of Object.entries(word_key_phrase_dict)) {
            let word_doc_ids = [];
            // Match the key phrases with title words and assign key phrase
            for (const key_phrase of key_phrases) {
                // // Collect the doc containing the key phrase
                const relevant_docs = docs.filter(d => {
                    const found = d['KeyPhrases'].find(kp => kp.toLowerCase() === key_phrase.toLowerCase());
                    if (found) {
                        return true;
                    }
                    return false;
                });
                for (const doc of relevant_docs) {
                    if (!word_doc_ids.includes(doc['DocId'])) {
                        word_doc_ids.push(doc['DocId']);
                    }
                }
            }
            dict[title_word] = word_doc_ids;
        }
        return dict;
    }

    // Create a dict of word and key phrases
    // Allocate the key phrases based on the words
    static create_word_key_phrases_dict(title_words, key_phrases) {
        let dict = {};
        // Initialise dict with an array
        for (const title_word of title_words) {
            dict[title_word] = [];
        }
        // Add 'others'
        dict['others'] = [];

        // Match the key phrases with title words and assign key phrase
        for (const key_phrase of key_phrases) {
            let is_found = false;
            for (const title_word of title_words) {
                if (key_phrase.toLowerCase().includes(title_word.toLowerCase())) {
                    dict[title_word].push(key_phrase);
                    is_found = true;
                }
            }
            // If not found, assign to 'misc'
            if (!is_found) {
                dict['others'].push(key_phrase);
            }
        }
        return dict;
    }


    // Collect top 5 words from a keyword cluster
    static get_top_5_words_from_keyword_group(keyword_group){
        const key_phrases = keyword_group['key-phrases'];
        try {
            let word_key_phrases = [];
            for (const key_phrase of key_phrases) {
                const words = key_phrase.toLowerCase().split(" ");
                for (const word of words) {
                    let word_key_phrase = word_key_phrases.find(w => w['word'] === word);
                    if(!word_key_phrase){
                        word_key_phrase = {
                            'word': word, 'phrase': []
                        };
                        word_key_phrases.push(word_key_phrase);
                    }
                    word_key_phrase['phrase'].push(key_phrase);
                }
            }
            // Sort words by the number of phrases
            word_key_phrases.sort((a, b) => b['phrase'].length - a['phrase'].length);
            // console.log(word_key_phrases);
            return word_key_phrases.map(w => w['word']).slice(0, 5);
        }catch (error) {
            console.error(error);
        }
    }

    // Collect the unique top 3 terms
    static get_top_terms(cluster_terms, n) {
        // Get top 3 term that do not overlap
        let top_terms = cluster_terms.slice(0, 1);
        let index = 1;
        while (top_terms.length < n && index < cluster_terms.length) {
            const candidate_term = cluster_terms[index];
            try {
                const c_term_words = candidate_term.split(" ");
                if (c_term_words.length >= 2) {
                    // Check if candidate_term overlap with existing top term
                    const overlap = top_terms.find(top_term => {
                        // Check if any cluster terms
                        const top_term_words = top_term.split(" ");
                        // Check two terms are different
                        // Split the terms into words and check if two term has overlapping word
                        // Check if two word list has intersection
                        const intersection = top_term_words.filter(term => c_term_words.includes(term));
                        if (intersection.length > 0) {
                            return true;
                        }
                        return false;
                    });
                    if (typeof overlap !== 'undefined') {
                        try {
                            if (overlap.split(" ").length !== candidate_term.split(" ").length) {
                                const overlap_index = top_terms.findIndex(t => t === overlap);
                                // Replace existing top term with candidate
                                top_terms[overlap_index] = candidate_term;
                            }// Otherwise, skip the term
                        } catch (error) {
                            console.error(error);
                            break;
                        }
                    } else {
                        top_terms.push(candidate_term);
                    }
                }
            } catch (error) {
                console.error(error);
                break;
            }
            index = index + 1;
        }
        // Check if the top_terms has n terms
        while (top_terms.length < n) {
            // Get the last term
            const last_term = top_terms[top_terms.length - 1];
            const last_term_index = cluster_terms.findIndex(t => t === last_term) + 1;
            top_terms.push(cluster_terms[last_term_index]);
        }


        return top_terms;
    }


}



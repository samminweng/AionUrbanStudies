// Display the instruction
function InstructionDialog(chart_type, auto_open){
    let instruction_text;
    if(chart_type === 'wordtree'){
        instruction_text = 'Clicking on a topic (green words) displays the articles associated within the topic.';
    }else{
        instruction_text = '1. <b>Mouse over a cluster (outer circle)</b> display top 10 topics within the cluster.<br>' +
            '2. <b>Clicking on a cluster</b> displays the articles within the cluster.';
    }

    $('#instruction').empty();
    $('#instruction').append($('<p>'+ instruction_text+'</p>'));
    $('#instruction').dialog({
        autoOpen: auto_open,
        modal: true,
        buttons: {
            Ok: function () {
                $(this).dialog("close");
            }
        }
    });
    $("#opener").on("click", function () {
        $("#instruction").dialog("open");
    });

}

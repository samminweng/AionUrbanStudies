// Display the instruction
function InstructionDialog(auto_open){
    let instruction_text = '1. <b>Mouse over a dot</b> to display top 5 topics of a cluster. <br>' +
            '2. <b>Click on a dot</b> to display the cluster, top topics and associated articles within the cluster.<br>' +
            '3. <b>Tick or Untick outlier box</b> to display the outliers on the chart.';

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

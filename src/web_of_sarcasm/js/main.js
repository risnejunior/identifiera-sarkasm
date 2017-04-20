var ajaxurl = "index.php"
var debug = true

/* event handlers ####################################################
####################################################################*/
$( document ).ready(function() {
	debug ? console.log("document load") : null
	var user_id = null 

	//user unknown
	if (localStorage.getItem('user_id') === null ||
		localStorage.getItem('nonce') === null ) {
		
		$("#intro").show();

	//user known
	} else {
		update_quiz('easy', 10);
	}
});

// register user
$("#name-form").keypress(function(e) {
	if(e.which == 13){
		$("#name-form").submit();
	}
});

$("#name-form").submit(function(event) {
	event.preventDefault();
	register_user();
});

$("#hard-link").click(function(e){
	$("li", "#navbar").removeClass("selected");
	$(this).addClass("selected");
	update_quiz('hard', 10);
});

$("#easy-link").click(function(e){
	$("li", "#navbar").removeClass("selected");
	$(this).addClass("selected");
	update_quiz('easy', 10);
});

/* funs #############################################################
####################################################################*/

function getScore(dataset, callback) {
	debug ? console.log("get score") : null
	var message = {
		'user_id': localStorage['user_id'],
		'action': 'tally_score',
		'dataset': dataset
	};

	send_ajax(message, success);
	$("#score").parent().fadeTo(10, 0)

	function success(data) {
		var tally = data.tally[0];		
		localStorage[data.dataset + '_tally'] = JSON.stringify(tally);
		var metrics = calculate_metrics(tally);
		$("#score").parent().fadeTo(300, 1, callback(metrics));
	}
}

function update_metrics(dataset) {
	getScore(dataset, callback)

	function callback(metrics) {
		debug ? console.log("metrics callback") : null
		$("#score-header").text(dataset + " quiz score:");		
		$("#count").html(metrics.count);		
		$("#accuracy").html(metrics.accuracy);
		$("#precision").html(metrics.precision);
		$("#recall").html(metrics.recall);
		$("#f1_score").html(metrics.f1_score);
		$("#aside").show();
		$("#score").show();
	}
}

function send_ajax(message, success_fun) {
	debug ? console.log("send ajax") : null
	var settings = {
	     "url": ajaxurl,
	     "timeout": 5000,
	     "type": "post",
	     "dataType": "json",
	     "data": message,
	     "complete": complete,
	     "error": error,
	     "success": success,
	     "context": this
	};
	 
	 //ajax call bound to deferred
	jQuery.ajax( settings );

	function success( data, textStatus, jqXHR ) {
		
		//serverside error 
	    if ( !data.type || "error" === data.type ) {
	        var err_message = "An error occurred!";
	        if ( 'errors' in data ) {
	            err_message = "Error: " + data.errors.join(',');
	        }
	        jQuery("#status").html( err_message ).show();

	        /*
	    	if (err_message.indexOf("User not found") == -1) {
	    		localstore.clear();
	    		location.reload();
	    	}
	    	*/
	    
	    //server side success: set quiz
	    } else {
	    	success_fun(data);
	    }
	}

	function error( jqXHR, textStatus, errorThrown ) {
		jQuery("#status").html( "Server error!" ).show();
	}

	function complete( jqXHR, textStatus ) {
		jQuery( this )
		    .parent()
		    .siblings( ".animation-container" )
		    .removeClass( "working-animation" );
	}
}

function register_user() {
	debug ? console.log("register user") : null
	var message = {
		'name': $( "input[name*='name']" ).first().val(),
		'action': 'register_user'
	};

	send_ajax(message, success);

	function success(data) {
		localStorage['user_id'] = data.id;
		localStorage['nonce'] = data.nonce;
		localStorage['name'] = data.name;

		update_quiz('easy', 10);
	}
}

function get_quiz(dataset, size, callback) {
	debug ? console.log("get quiz") : null

	// check if quiz stored
	if (!(dataset in localStorage)) {
		debug ? console.log(dataset + " NOT in localstore") : null
	
		var message = {
			'action': 'get_quiz',
			'dataset': dataset,
			'user_id': parseInt(localStorage['user_id']),
			'nonce': localStorage['nonce'],
			'size': size
		};

		send_ajax(message, success);
		$("#quiz").fadeTo(300, 0);

	} else {
		debug ? console.log(dataset + " IS in localstore") : null
		quiz = JSON.parse(localStorage[dataset]);		
		callback(quiz);		
	}


	function success(data) {
		$("#quiz").fadeTo(300, 1);

		if (data.quiz.length < 1) {
			debug ? console.log("Empty quiz returned!") : null
			location.reload();
		}

		localStorage[data.dataset] = JSON.stringify(data.quiz);
		callback(data.quiz);
	}

	update_metrics(dataset);
}

function add_quiz_html(quiz, container, quiz_name) {
	debug ? console.log("add quiz html: " + quiz_name) : null
		
	container.empty();
	if (quiz_name == "hard") {
		container.removeClass("easy")
		container.addClass("hard")
	} else {
		container.removeClass("hard")
		container.addClass("easy")
	}

	var answers = get_answers();

	for (var i = 0; i < quiz.length; i++) {		
		question = quiz[i];
		//console.log(question.sample_text);

		var sectionElement = $("<section></section>")
			.attr("id", question.id)
			.text(question.sample_text)
			.addClass("question");

		var buttonPart = $("<div></div>")
			.addClass("button-part")
			.append("<button class='pos button'>Sarcastic</button>")
			.append("<button class='neg button'>Other</button>");

		if (question.id in answers && answers[question.id] == true) {
			$('.pos', buttonPart).addClass("selected");
		} else if (question.id in answers && answers[question.id] == false) {
			$('.neg', buttonPart).addClass("selected");
		}

		sectionElement.append(buttonPart);
		$(container).append(sectionElement);

	}

	$("button", ".button-part").click(function(event) {
		var question_id = $(this).parents("section").first().attr("id");		
		var isPos = $(this).hasClass("pos");
		
		if ( !$(this).hasClass('selected') ) {
			$(this).addClass('selected');
		}

		$(this).siblings('.button:first').removeClass('selected');

		set_answer(question_id, isPos, this);		
	});

	jQuery("#aside, #navbar").show();	
	jQuery("#intro").hide();
	jQuery("container").show();	
}

function set_answer(question_id, isPos, that) {
	debug ? console.log("Set answer: " + question_id + " " + isPos) : null
	user_id = localStorage.getItem('user_id')
	nonce = localStorage.getItem('nonce')
	var message = {
		'action': 'save_answer',
		'nonce': nonce,
		'question_id': parseInt(question_id),
		'user_id': parseInt(user_id),		
		'answer': isPos ? 1 : 0
	};

	send_ajax(message, success)
	$(that).parents("section:first").addClass("working-animation");

	function success(data) {
		$(that).parents("section:first").removeClass("working-animation");
		answers = get_answers()
		answers[question_id] = isPos;		
    	localStorage['answers'] = JSON.stringify(answers);
    	var done = check_if_done(answers);
    	if (done) {
    		replace_quiz();

    	}
	}
}

function replace_quiz() {
	debug ? console.log("replace quiz") : null
	container = $("#quiz");
	if (container.hasClass("easy")) {
		delete localStorage['poria-balanced'];
		update_quiz("easy", 10);
	} else {
		delete localStorage['poria-ratio'];
		update_quiz("hard", 10);
	}
}

function check_if_done(answers) {
	var ids = [];
	var allIn = true;
	$("section", "#quiz").each(function(){
		//ids.push(this.id);
		if (!(this.id in answers)) {
			allIn = false;
		}
	});

	return allIn;
}

function get_answers() {
	var answers = {};
	if ('answers' in localStorage) {
		answers = JSON.parse(localStorage['answers']);
	} 
	return answers;
}

function update_quiz(quiz_name, size) {
	debug ? console.log("update_quiz") : null
	//debug!
	size = 3;

	var dataset = null;
	var container = $("#quiz");

	if (quiz_name == 'hard') {		
		dataset = 'poria-ratio';		
	} else {
		dataset = 'poria-balanced';
	}

	 get_quiz(dataset, size, callback);	 

	function callback(quiz) {
		debug ? console.log("update quiz callback") : null
		add_quiz_html(quiz, container, quiz_name);		
	}
	
}

function calculate_metrics(tally) {	
	var metrics = {
		'accuracy': 0, 
		'precision': 0, 
		'recall': 0, 
		'f1_score': 0, 
		'count': 0
	};

	if (tally && tally.answer_count > 0) {
		var count = parseInt(tally.answer_count);
		var tp = parseInt(tally.tp); 
		var tn = parseInt(tally.tn); 
		var fp = parseInt(tally.fp); 
		var fn = parseInt(tally.fn);
		var accuracy = count > 0 ? ((tp + tn) / count) : 0
		var precision = (tp + fp) > 0 ? (tp / (tp + fp)) : 0
		var recall = (tp + fn) > 0 ? (tp / (tp + fn)) : 0
		var f1_score =  (precision + recall) > 0 ? 2*((precision * recall) / (precision + recall )) : 0
		metrics.count = count;
		metrics.accuracy = Math.round(accuracy * 100) / 100;
		metrics.f1_score = Math.round(f1_score * 100) / 100;
		metrics.precision = Math.round(precision * 100) / 100;
		metrics.recall = Math.round(recall * 100) / 100;
	}
	
	return metrics;
}
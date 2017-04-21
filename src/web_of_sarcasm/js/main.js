var ajaxurl = "index.php";
var debug = true;
var model = null;
var quiz_size = 3;
var datasetNames = {
	'easy': 'poria-balanced',
	'hard': 'poria-ratio'
}
var def_quiz = datasetNames['easy'];

/* data structs and util funs ########################################
####################################################################*/
var objectStore = new function() {
	this.set = function(store, obj) {
		localStorage[store] = JSON.stringify(obj);
	};
	this.get = function(store) {
		var item = null;
		if (store in localStorage) {
			item = JSON.parse(localStorage[store]);
		}
		
		return item;
	};
	this.update = function(store, key, val) {
		var obj = this.get(store);
		if (obj != null) {
			obj.key = val;
			this.set(store, obj);	
		}
		
		return obj;
	};
}

function Quiz (quiz) {
	this.questions = quiz.questions;
	this.metrics = quiz.metrics;
	this.answers = {};
}

function Model () {	
	this.user = {
		id: null,
		name: null,
		nonce: null
	};
	this.quizes = {};
}

function send_ajax(message, cb_success, cb_error, cb_complete) {
	debug ? console.log("send ajax") : null
	var settings = {
	     "url": ajaxurl,
	     "timeout": 3000,
	     "type": "post",
	     "dataType": "json",
	     "data": message,
	     "complete": cb_complete,
	     "error": error,
	     "success": success,
	     "context": this
	};
	 
	 //ajax call bound to deferred
	jQuery.ajax( settings );

	function success( data, textStatus, jqXHR ) {
		
		//serverside error 
	    if ( !data.type || "error" === data.type ) {
	        var err_message = "Server message not recognized!";
	        if ( 'errors' in data ) {
	            err_message = data.errors.join(', ');
	        }

	        cb_error(err_message);	        
	    //server side success: return data
	    } else {
	    	cb_success(data);
	    }
	}

	function error( jqXHR, textStatus, errorThrown ) {
		var err_message = null;
		if (jqXHR.readyState == 4) {
            err_message = "Http Error";
        } else if (jqXHR.readyState == 0) {
            err_message = "Network error";            
        } else {
            err_message = "Unhandled error, " + textStatus;
        }

		cb_error(err_message);
	}
}
/* controller #######################################################
####################################################################*/
var quiz_container = $("#quiz");
var metrics_container = $("#metrics");
var intro_container = $("#intro");
var navbar = $('#navbar');
var aside = $('#aside');

$( document ).ready(function() {
	is_new_user = model_init();
	
	if (is_new_user) {
		quiz_container.hide()
		intro_container.show()
	} else {
		ctrl_refresh_quiz(def_quiz, false);
	}
});

// register user keypress submit
$("#name-form").keypress(function(e) {
	if(e.which == 13){
		$(this).submit();
	}
});

// register user
$("#name-form").submit(function(event) {
	event.preventDefault();
	var name = $( "input[name*='name']" ).first().val();
	model_register_user(name, cb_view_success, cb_view_error);

	function cb_view_success() {		
		ctrl_refresh_quiz(def_quiz, false);
	} 

	function cb_view_error(err_message) {
		view_display_error(err_message);
	}
});

$("#hard-link").click(function(e){
	var dataset = datasetNames['hard'];
	$("li", "#navbar").removeClass("selected");
	$(this).addClass("selected");
		
	ctrl_refresh_quiz(dataset, false);
});

$("#easy-link").click(function(e){
	var dataset = datasetNames['easy'];	
	$("li", "#navbar").removeClass("selected");
	$(this).addClass("selected");
	
	ctrl_refresh_quiz(dataset, false);
});


$("#quiz").on("click", "button", function(event) {
	var section = $(this).parents("section").first();
	var question_id = section.attr("id");		
	var answer = $(this).hasClass("pos") ? 1 : 0;
	var button = $(this);
	var dataset = section.attr("dataset");

	if ( !$(this).hasClass('selected') ) {
		$(this).addClass('selected');
	}
	$(this).siblings('.button:first').removeClass('selected');

	section.addClass("working-animation");
	model_set_answer(dataset, question_id, answer, cb_error, cb_complete);

	function cb_complete(allAnswered) {		
		section.removeClass("working-animation");

		if (allAnswered) {
			ctrl_refresh_quiz(dataset, true, quiz_container, metrics_container)
		}
	}

	function cb_error(err_message) {
		debug ? console.log(err_message) : null
		button.removeClass('selected');
		view_display_error(err_message);
	}
});

$("#error").on('click', 'p', function(e) {
	$("#error").hide();
	$("#main-content").slideDown();
});

$("#logout a").click(function(e) {
	e.preventDefault();
	localStorage.clear();
	document.location.reload();
});

function ctrl_refresh_quiz(dataset, replace) {
	intro_container.hide();
	navbar.show();
	aside.show();
	quiz_container.fadeTo(10, 0);
	metrics_container.fadeTo(10, 0);

	model_get_quiz(dataset, quiz_size, replace, cb_success, cb_error, cb_complete)

	function cb_success(quiz) {
		view_update_quiz(dataset, quiz.questions, quiz.answers);
		metrics = model_calculate_metrics(quiz.metrics);
		view_update_metrics(metrics, dataset);
	}

	function cb_error(err_message) {
		debug ? console.log("refresh quiz, error getting quiz") : null
		view_display_error(err_message);
	}

	function cb_complete() {
		quiz_container.fadeTo(300, 1);
		metrics_container.fadeTo(300, 1);
	}
	
}

/* view funs #############################################################
####################################################################*/

function view_update_quiz(dataset, questions, answers) {
	quiz_container.empty();

	for (var i = 0; i < questions.length; i++) {				
		question = questions[i];
		//console.log(question.sample_text);

		var sectionElement = $("<section></section>")
			.attr("id", question.id)
			.text(question.sample_text)
			.addClass("question")
			.attr("dataset", dataset);

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
		quiz_container.append(sectionElement);

	}
}

function view_update_metrics(metrics, dataset) {
	$("#score-header").text(dataset + " quiz score:");	
	$("#count").html(metrics.count);		
	$("#accuracy").html(metrics.accuracy);
	$("#precision").html(metrics.precision);
	$("#recall").html(metrics.recall);
	$("#f1_score").html(metrics.f1_score);
	$("#aside").show();
	$("#score").show();
}

function view_display_error(err_message) {
	$("#main-content").slideUp();
	$("#error h2:first").html("ERROR: " + err_message);
	$("#error p").html("(CLICK TO CLEAR)");
	$("#error").show();

}


/* model funs #############################################################
####################################################################*/
function model_init() {
	new_user = false 
	model = objectStore.get("model");

	//new user
	if (model === null || model.user.id === null) {
		model = new Model();
		new_user = true;
		objectStore.set("model", model);		
	//known user
	} 
	return new_user;
}

function model_register_user(name, cb_view_success, cb_view_error, cb_view_complete) {
	debug ? console.log("register user") : null
	var message = {
		'name': name,
		'action': 'register_user'
	};

	send_ajax(message, cb_success, cb_view_error, cb_view_complete);

	function cb_success(data) {
		model.user.id = data.id
		model.user.nonce = data.nonce;
		model.user.name = data.name;
		objectStore.set("model", model);

		cb_view_success();
	}
}

function model_get_quiz(dataset, size, replace, cb_view_success, cb_view_error, cb_view_complete) {
	var quiz = model.quizes[dataset];
	
	if (quiz && !replace) {
		// quiz in localstore
		cb_view_success(quiz);
		cb_view_complete();

	} else {
		//not in localstore
		var message = {
			'action': 'get_quiz',
			'dataset': dataset,
			'user_id': model.user.id,
			'nonce': model.user.nonce,
			'size': size
		};

		send_ajax(message, cb_success, cb_view_error, cb_view_complete);
	}

	function cb_success(data) {		
		quiz = new Quiz(data.quiz);
		model.quizes[dataset] = quiz;
		objectStore.set("model", model);

		cb_view_success(quiz);
	}
}

function model_set_answer(dataset, question_id, answer, cb_view_error, cb_view_complete) {
	var message = {
		'action': 'save_answer',
		'nonce': model.user.nonce,
		'question_id': question_id,
		'user_id': model.user.id,		
		'answer': answer
	};
	send_ajax(message, cb_success, cb_view_error, cb_complete)

	function cb_success(data) {
		model.quizes[dataset].answers[question_id] = answer;
		objectStore.set("model", model);
	}

	function cb_complete() {
		var allAnswered = false;
		var quiz = model.quizes[dataset];
		if (Object.keys(quiz.answers).length >= quiz.questions.length) {
			allAnswered = true;
		}
		cb_view_complete(allAnswered);
	}
}


function model_calculate_metrics(tally) {	
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
def enhanced_knowledge_transfer(teacher_model, student_model, task, logger):
    """Copy as many weights as possible from teacher to student."""
    try:
        logger.info("Performing enhanced knowledge transfer")
        
        # Copy embeddings (common to all models)
        if hasattr(teacher_model, "distilbert") and hasattr(student_model, "distilbert"):
            # Copy word embeddings
            student_model.distilbert.embeddings.word_embeddings.weight.data = \
                teacher_model.distilbert.embeddings.word_embeddings.weight.data.clone()
            logger.info("Copied word embeddings")
            
            # Copy position embeddings if they exist and have the same size
            if hasattr(teacher_model.distilbert.embeddings, "position_embeddings") and \
               hasattr(student_model.distilbert.embeddings, "position_embeddings") and \
               teacher_model.distilbert.embeddings.position_embeddings.weight.shape == \
               student_model.distilbert.embeddings.position_embeddings.weight.shape:
                student_model.distilbert.embeddings.position_embeddings.weight.data = \
                    teacher_model.distilbert.embeddings.position_embeddings.weight.data.clone()
                logger.info("Copied position embeddings")
            
            # Copy as many transformer blocks as possible
            teacher_layers = teacher_model.distilbert.transformer.layer
            student_layers = student_model.distilbert.transformer.layer
            
            # Copy weights from the first N layers (where N is the number of student layers)
            for i in range(min(len(student_layers), len(teacher_layers))):
                # Copy attention weights
                if hasattr(teacher_layers[i], "attention") and hasattr(student_layers[i], "attention"):
                    # Copy query, key, value weights
                    if hasattr(teacher_layers[i].attention, "q_lin") and hasattr(student_layers[i].attention, "q_lin"):
                        if teacher_layers[i].attention.q_lin.weight.shape == student_layers[i].attention.q_lin.weight.shape:
                            student_layers[i].attention.q_lin.weight.data = teacher_layers[i].attention.q_lin.weight.data.clone()
                            student_layers[i].attention.q_lin.bias.data = teacher_layers[i].attention.q_lin.bias.data.clone()
                            logger.info(f"Copied attention Q weights for layer {i}")
                    
                    if hasattr(teacher_layers[i].attention, "k_lin") and hasattr(student_layers[i].attention, "k_lin"):
                        if teacher_layers[i].attention.k_lin.weight.shape == student_layers[i].attention.k_lin.weight.shape:
                            student_layers[i].attention.k_lin.weight.data = teacher_layers[i].attention.k_lin.weight.data.clone()
                            student_layers[i].attention.k_lin.bias.data = teacher_layers[i].attention.k_lin.bias.data.clone()
                            logger.info(f"Copied attention K weights for layer {i}")
                    
                    if hasattr(teacher_layers[i].attention, "v_lin") and hasattr(student_layers[i].attention, "v_lin"):
                        if teacher_layers[i].attention.v_lin.weight.shape == student_layers[i].attention.v_lin.weight.shape:
                            student_layers[i].attention.v_lin.weight.data = teacher_layers[i].attention.v_lin.weight.data.clone()
                            student_layers[i].attention.v_lin.bias.data = teacher_layers[i].attention.v_lin.bias.data.clone()
                            logger.info(f"Copied attention V weights for layer {i}")
                
                # Copy FFN weights
                if hasattr(teacher_layers[i], "ffn") and hasattr(student_layers[i], "ffn"):
                    if hasattr(teacher_layers[i].ffn, "lin1") and hasattr(student_layers[i].ffn, "lin1"):
                        if teacher_layers[i].ffn.lin1.weight.shape == student_layers[i].ffn.lin1.weight.shape:
                            student_layers[i].ffn.lin1.weight.data = teacher_layers[i].ffn.lin1.weight.data.clone()
                            student_layers[i].ffn.lin1.bias.data = teacher_layers[i].ffn.lin1.bias.data.clone()
                            logger.info(f"Copied FFN1 weights for layer {i}")
                    
                    if hasattr(teacher_layers[i].ffn, "lin2") and hasattr(student_layers[i].ffn, "lin2"):
                        if teacher_layers[i].ffn.lin2.weight.shape == student_layers[i].ffn.lin2.weight.shape:
                            student_layers[i].ffn.lin2.weight.data = teacher_layers[i].ffn.lin2.weight.data.clone()
                            student_layers[i].ffn.lin2.bias.data = teacher_layers[i].ffn.lin2.bias.data.clone()
                            logger.info(f"Copied FFN2 weights for layer {i}")
        
        # Handle BERT models (different architecture)
        elif hasattr(teacher_model, "bert") and hasattr(student_model, "bert"):
            # Copy word embeddings
            if teacher_model.bert.embeddings.word_embeddings.weight.shape == student_model.bert.embeddings.word_embeddings.weight.shape:
                student_model.bert.embeddings.word_embeddings.weight.data = \
                    teacher_model.bert.embeddings.word_embeddings.weight.data.clone()
                logger.info("Copied BERT word embeddings")
            
            # Copy position embeddings
            if hasattr(teacher_model.bert.embeddings, "position_embeddings") and \
               hasattr(student_model.bert.embeddings, "position_embeddings") and \
               teacher_model.bert.embeddings.position_embeddings.weight.shape == \
               student_model.bert.embeddings.position_embeddings.weight.shape:
                student_model.bert.embeddings.position_embeddings.weight.data = \
                    teacher_model.bert.embeddings.position_embeddings.weight.data.clone()
                logger.info("Copied BERT position embeddings")
            
            # Copy as many transformer blocks as possible
            teacher_layers = teacher_model.bert.encoder.layer
            student_layers = student_model.bert.encoder.layer
            
            # Copy weights from the first N layers (where N is the number of student layers)
            for i in range(min(len(student_layers), len(teacher_layers))):
                # Copy attention weights
                if hasattr(teacher_layers[i].attention, "self") and hasattr(student_layers[i].attention, "self"):
                    # Copy query, key, value weights
                    if teacher_layers[i].attention.self.query.weight.shape == student_layers[i].attention.self.query.weight.shape:
                        student_layers[i].attention.self.query.weight.data = teacher_layers[i].attention.self.query.weight.data.clone()
                        student_layers[i].attention.self.query.bias.data = teacher_layers[i].attention.self.query.bias.data.clone()
                        logger.info(f"Copied BERT attention query weights for layer {i}")
                    
                    if teacher_layers[i].attention.self.key.weight.shape == student_layers[i].attention.self.key.weight.shape:
                        student_layers[i].attention.self.key.weight.data = teacher_layers[i].attention.self.key.weight.data.clone()
                        student_layers[i].attention.self.key.bias.data = teacher_layers[i].attention.self.key.bias.data.clone()
                        logger.info(f"Copied BERT attention key weights for layer {i}")
                    
                    if teacher_layers[i].attention.self.value.weight.shape == student_layers[i].attention.self.value.weight.shape:
                        student_layers[i].attention.self.value.weight.data = teacher_layers[i].attention.self.value.weight.data.clone()
                        student_layers[i].attention.self.value.bias.data = teacher_layers[i].attention.self.value.bias.data.clone()
                        logger.info(f"Copied BERT attention value weights for layer {i}")
                
                # Copy FFN weights
                if hasattr(teacher_layers[i], "intermediate") and hasattr(student_layers[i], "intermediate"):
                    if teacher_layers[i].intermediate.dense.weight.shape == student_layers[i].intermediate.dense.weight.shape:
                        student_layers[i].intermediate.dense.weight.data = teacher_layers[i].intermediate.dense.weight.data.clone()
                        student_layers[i].intermediate.dense.bias.data = teacher_layers[i].intermediate.dense.bias.data.clone()
                        logger.info(f"Copied BERT intermediate weights for layer {i}")
                
                if hasattr(teacher_layers[i], "output") and hasattr(student_layers[i], "output"):
                    if hasattr(teacher_layers[i].output, "dense") and hasattr(student_layers[i].output, "dense"):
                        if teacher_layers[i].output.dense.weight.shape == student_layers[i].output.dense.weight.shape:
                            student_layers[i].output.dense.weight.data = teacher_layers[i].output.dense.weight.data.clone()
                            student_layers[i].output.dense.bias.data = teacher_layers[i].output.dense.bias.data.clone()
                            logger.info(f"Copied BERT output weights for layer {i}")
        
        # Task-specific output layer transfer
        if task == "sequence-classification" or task == "text-classification":
            if hasattr(teacher_model, "classifier") and hasattr(student_model, "classifier"):
                if teacher_model.classifier.out_features == student_model.classifier.out_features:
                    student_model.classifier.weight.data = teacher_model.classifier.weight.data.clone()
                    student_model.classifier.bias.data = teacher_model.classifier.bias.data.clone()
                    logger.info("Copied classifier weights")
                else:
                    logger.info(f"Classifier dimensions don't match: teacher {teacher_model.classifier.out_features}, student {student_model.classifier.out_features}")
            else:
                logger.info("Models don't have classifier attribute")
        
        elif task == "token-classification":
            if hasattr(teacher_model, "classifier") and hasattr(student_model, "classifier"):
                if teacher_model.classifier.out_features == student_model.classifier.out_features:
                    student_model.classifier.weight.data = teacher_model.classifier.weight.data.clone()
                    student_model.classifier.bias.data = teacher_model.classifier.bias.data.clone()
                    logger.info("Copied token classifier weights")
                else:
                    logger.info(f"Token classifier dimensions don't match: teacher {teacher_model.classifier.out_features}, student {student_model.classifier.out_features}")
            else:
                logger.info("Models don't have classifier attribute for token classification")
        
        elif task == "question-answering":
            if hasattr(teacher_model, "qa_outputs") and hasattr(student_model, "qa_outputs"):
                if teacher_model.qa_outputs.weight.shape == student_model.qa_outputs.weight.shape:
                    student_model.qa_outputs.weight.data = teacher_model.qa_outputs.weight.data.clone()
                    student_model.qa_outputs.bias.data = teacher_model.qa_outputs.bias.data.clone()
                    logger.info("Copied QA output weights")
                else:
                    logger.info(f"QA output dimensions don't match: teacher {teacher_model.qa_outputs.weight.shape}, student {student_model.qa_outputs.weight.shape}")
            else:
                logger.info("Models don't have qa_outputs attribute")
        
        elif task == "masked-lm" or task == "fill-mask":
            # Try different MLM head architectures
            if hasattr(teacher_model, "vocab_projector") and hasattr(student_model, "vocab_projector"):
                if teacher_model.vocab_projector.weight.shape == student_model.vocab_projector.weight.shape:
                    student_model.vocab_projector.weight.data = teacher_model.vocab_projector.weight.data.clone()
                    student_model.vocab_projector.bias.data = teacher_model.vocab_projector.bias.data.clone()
                    logger.info("Copied vocab projector weights")
                else:
                    logger.info(f"Vocab projector dimensions don't match: teacher {teacher_model.vocab_projector.weight.shape}, student {student_model.vocab_projector.weight.shape}")
            elif hasattr(teacher_model, "cls") and hasattr(student_model, "cls"):
                if hasattr(teacher_model.cls, "predictions") and hasattr(student_model.cls, "predictions"):
                    if hasattr(teacher_model.cls.predictions, "decoder") and hasattr(student_model.cls.predictions, "decoder"):
                        if teacher_model.cls.predictions.decoder.weight.shape == student_model.cls.predictions.decoder.weight.shape:
                            student_model.cls.predictions.decoder.weight.data = teacher_model.cls.predictions.decoder.weight.data.clone()
                            if hasattr(teacher_model.cls.predictions.decoder, "bias") and hasattr(student_model.cls.predictions.decoder, "bias"):
                                student_model.cls.predictions.decoder.bias.data = teacher_model.cls.predictions.decoder.bias.data.clone()
                            logger.info("Copied MLM decoder weights")
                        else:
                            logger.info(f"MLM decoder dimensions don't match: teacher {teacher_model.cls.predictions.decoder.weight.shape}, student {student_model.cls.predictions.decoder.weight.shape}")
                    else:
                        logger.info("Models don't have decoder attribute for MLM")
                else:
                    logger.info("Models don't have predictions attribute for MLM")
            else:
                logger.info("Models don't have vocab_projector or cls attribute for MLM")
        
        logger.info("Enhanced knowledge transfer completed")
        return True
    except Exception as e:
        logger.error(f"Error during enhanced knowledge transfer: {e}")
        logger.error(traceback.format_exc())
        return False

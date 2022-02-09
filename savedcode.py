def trainer(model, train_loader, valid_loader=None):
    model.to(device)

    def run_epoch(split):
        is_train = split == 'train' 
        model.train(is_train)
        loader = train_loader if is_train else valid_loader

        avg_loss = 0
        pbar = tqdm(enumerate(loader), total=len(loader))
        for step, batch in pbar:  
            batch = [i.to(device) for i in batch]
            seq, mask, labels = batch
            
            with torch.set_grad_enabled(is_train): 
                preds = model(seq, mask)
                loss = cost(preds, labels)
                avg_loss += loss.item() / len(loader)

            if is_train:
                model.zero_grad() 
                loss.backward() 
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) 
                optimizer.step()

            pbar.set_description(f"epoch: {e+1}, loss: {loss.item():.3f}, avg: {avg_loss:.2f}")     
        return avg_loss

    best_loss = float('inf')
    cost = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    
    for e in range(EPOCHS):
        train_loss = run_epoch('train')
        valid_loss = run_epoch('valid') 

        if valid_loss < best_loss:
            best_loss = valid_loss
            torch.save(model.state_dict(), 'best.bert.classifier')



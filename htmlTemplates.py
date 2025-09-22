css = '''
<style>
/* Chat container styling */
.chat-message {
    padding: 1.5rem;
    border-radius: 0.5rem;
    margin-bottom: 1rem;
    display: flex;
    align-items: center;
    font-family: Arial, sans-serif;
}

/* User message styling */
.chat-message.user {
    background-color: #2b313e;
    color: #fff;
}

/* Bot message styling */
.chat-message.bot {
    background-color: #475063;
    color: #fff;
}

/* Avatar container */
.chat-message .avatar {
    width: 60px;
    height: 60px;
    margin-right: 1rem;
    flex-shrink: 0;
}

/* Avatar image */
.chat-message .avatar img {
    width: 100%;
    height: 100%;
    border-radius: 50%;
    object-fit: cover;
    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.2);
}

/* Message text */
.chat-message .message {
    flex-grow: 1;
    font-size: 1rem;
    line-height: 1.5;
    word-wrap: break-word;
}
</style>
'''

bot_template = '''
<div class="chat-message bot">
    <div class="avatar">
        <img src="https://imgcdn.stablediffusionweb.com/2024/4/22/48aa42be-d83b-4f60-a975-db643dcedf2c.jpg" alt="Bot Avatar">
    </div>
    <div class="message">{{MSG}}</div>
</div>
'''

user_template = '''
<div class="chat-message user">
    <div class="avatar">
        <img src="https://upload.wikimedia.org/wikipedia/commons/9/99/Sample_User_Icon.png" alt="User Avatar">
    </div>
    <div class="message">{{MSG}}</div>
</div>
'''

VIRTUAL_SCROLL_JS = """
async function autoScroll() {
    await new Promise((resolve) => {
        let totalHeight = 0;
        const distance = 500;
        const timer = setInterval(() => {
            window.scrollBy(0, distance);
            totalHeight += distance;

            if(totalHeight >= document.body.scrollHeight){
                clearInterval(timer);
                resolve();
            }
        }, 300);
    });
}
autoScroll();
"""

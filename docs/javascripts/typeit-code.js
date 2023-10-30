const codeTypingActivateClass = "code-typing";
let codeTypingInstances = [];

function createCodeTypings() {
    document
        .querySelectorAll(`.${codeTypingActivateClass} .highlight`)
        .forEach(node => {
            let codeBlock = node.getElementsByTagName("code")[0];
            const codeBlockStrings = codeBlock.innerHTML;
            let newCodeBlock = document.createElement("div")
            let codeBlockSize = codeBlock.getClientRects()[0];
            codeBlock.style.height = codeBlockSize.height + "px";
            codeBlock.style.width = codeBlockSize.width + "px";
            codeBlock.innerHTML = "";

            codeBlock.appendChild(newCodeBlock);

            const typeIt = new TypeIt(
                newCodeBlock,
                {
                    strings: codeBlockStrings,
                    speed: 2,
                    waitUntilVisible: true,
                    cursor: false,
                    afterComplete: async (instance) => {
                        instance.getElement().parentElement.style.removeProperty("height")
                        instance.getElement().parentElement.style.removeProperty("width")
                    },
                }
            );
            console.log(typeIt);
            codeTypingInstances.push(typeIt);
        });
}


function loadCodeTypings() {
    codeTypingInstances.filter(termynal => {
        termynal.go();
    });
}

window.addEventListener("scroll", loadCodeTypings);
createCodeTypings();
loadCodeTypings();
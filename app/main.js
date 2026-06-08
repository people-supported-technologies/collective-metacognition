// D3 force-directed viewer for attributed causal maps
// Loads graph_data.json, renders fixed node universe, filters resize nodes/edges

(async function() {
    const data = await d3.json("graph_data.json");
    if (!data) { console.error("Failed to load graph_data.json"); return; }

    const { nodes, raw_edges, metadata } = data;

    // Populate filter dropdowns
    populateSelect("filter-round", metadata.rounds.map(r => ({ value: r, label: `Round ${r}` })));
    populateSelect("filter-table", metadata.tables.map(t => ({ value: t, label: t.slice(0, 8) })));
    populateSelect("filter-speaker", metadata.speakers.map(s => ({ value: s, label: s.slice(0, 12) })));

    // Stats
    document.getElementById("stat-nodes").textContent = nodes.length;
    document.getElementById("stat-edges").textContent = raw_edges.length;

    // SVG setup
    const container = document.getElementById("graph-container");
    const width = container.clientWidth;
    const height = container.clientHeight;

    const svg = d3.select("#graph-container")
        .append("svg")
        .attr("width", width)
        .attr("height", height);

    // Arrow marker
    svg.append("defs").append("marker")
        .attr("id", "arrowhead")
        .attr("viewBox", "0 -5 10 10")
        .attr("refX", 20)
        .attr("refY", 0)
        .attr("markerWidth", 6)
        .attr("markerHeight", 6)
        .attr("orient", "auto")
        .append("path")
        .attr("d", "M0,-5L10,0L0,5")
        .attr("class", "arrow-marker");

    const g = svg.append("g");

    // Zoom
    const zoom = d3.zoom().scaleExtent([0.1, 5]).on("zoom", (e) => g.attr("transform", e.transform));
    svg.call(zoom);

    // Build node/link structures for D3
    const nodeMap = new Map(nodes.map(n => [n.node_id, { ...n, x: width/2, y: height/2 }]));
    const nodeArray = Array.from(nodeMap.values());

    // Force simulation (all nodes always present)
    const simulation = d3.forceSimulation(nodeArray)
        .force("link", d3.forceLink().id(d => d.node_id).distance(80))
        .force("charge", d3.forceManyBody().strength(-50))
        .force("center", d3.forceCenter(width / 2, height / 2))
        .force("collision", d3.forceCollide().radius(20))
        .force("x", d3.forceX(width / 2).strength(0.05))
        .force("y", d3.forceY(height / 2).strength(0.05));

    // Edge elements
    const linkGroup = g.append("g").attr("class", "links");
    const labelGroup = g.append("g").attr("class", "edge-labels");
    const nodeGroup = g.append("g").attr("class", "nodes");
    const textGroup = g.append("g").attr("class", "node-texts");

    // Tooltip & Modal refs
    const tooltip = document.getElementById("tooltip");
    const modalOverlay = document.getElementById("modal-overlay");
    const modal = document.getElementById("modal");

    modalOverlay.addEventListener("click", (e) => {
        if (e.target === modalOverlay) closeModal();
    });
    document.addEventListener("keydown", (e) => { if (e.key === "Escape") closeModal(); });

    function showTooltip(event, html) {
        tooltip.innerHTML = html;
        tooltip.style.display = "block";
        tooltip.style.left = (event.pageX + 12) + "px";
        tooltip.style.top = (event.pageY - 10) + "px";
    }
    function hideTooltip() { tooltip.style.display = "none"; }
    function moveTooltip(event) {
        tooltip.style.left = (event.pageX + 12) + "px";
        tooltip.style.top = (event.pageY - 10) + "px";
    }

    function openModal(html) {
        modal.innerHTML = html;
        modalOverlay.classList.add("active");
    }
    function closeModal() { modalOverlay.classList.remove("active"); }

    function getFilteredEdges() {
        const round = document.getElementById("filter-round").value;
        const table = document.getElementById("filter-table").value;
        const speaker = document.getElementById("filter-speaker").value;
        const minConf = parseFloat(document.getElementById("filter-confidence").value);
        const stance = document.getElementById("filter-stance").value;
        const showInferred = document.getElementById("toggle-inferred").checked;

        return raw_edges.filter(e => {
            if (round !== "all" && e.round_id !== round) return false;
            if (table !== "all" && e.table_id !== table) return false;
            if (speaker !== "all" && e.participant_id !== speaker) return false;
            if (e.confidence < minConf) return false;
            if (stance !== "all" && e.stance !== stance) return false;
            if (!showInferred && e.explicitness === "inferred") return false;
            return true;
        });
    }

    function aggregateFilteredEdges(filtered) {
        const groups = new Map();
        for (const e of filtered) {
            const key = `${e.source_node_id}|${e.target_node_id}|${e.relation}`;
            if (!groups.has(key)) groups.set(key, []);
            groups.get(key).push(e);
        }
        return Array.from(groups.entries()).map(([key, edges]) => {
            const [source, target, relation] = key.split("|");
            const speakers = new Set(edges.map(e => e.participant_id));
            const confs = edges.map(e => e.confidence);
            const hasInferred = edges.some(e => e.explicitness === "inferred" || e.explicitness === "near_explicit");
            return {
                source, target, relation,
                count: edges.length,
                unique_speakers: speakers.size,
                mean_confidence: confs.reduce((a, b) => a + b, 0) / confs.length,
                has_inferred: hasInferred,
                evidence: edges.map(e => ({ text: e.evidence_text, speaker: e.speaker, stance: e.stance })),
            };
        });
    }

    function update() {
        const filtered = getFilteredEdges();
        const aggEdges = aggregateFilteredEdges(filtered);
        const hideIsolated = document.getElementById("toggle-isolated").checked;
        const showLabels = document.getElementById("toggle-labels").checked;

        document.getElementById("stat-visible").textContent = filtered.length;

        // Compute node weights
        const nodeWeights = new Map(nodes.map(n => [n.node_id, 0]));
        for (const e of filtered) {
            nodeWeights.set(e.source_node_id, (nodeWeights.get(e.source_node_id) || 0) + 1);
            nodeWeights.set(e.target_node_id, (nodeWeights.get(e.target_node_id) || 0) + 1);
        }

        // Links
        const links = aggEdges.map(e => ({
            source: e.source,
            target: e.target,
            ...e,
        }));

        simulation.force("link").links(links);

        // Update links
        const linkSel = linkGroup.selectAll("line").data(links, d => `${d.source.node_id || d.source}|${d.target.node_id || d.target}|${d.relation}`);
        linkSel.exit().remove();
        const linkEnter = linkSel.enter().append("line")
            .attr("class", "edge-line")
            .attr("cursor", "pointer")
            .attr("marker-end", "url(#arrowhead)")
            .on("mouseenter", (event, d) => {
                showTooltip(event, `
                    <div class="tt-rel">${d.relation}</div>
                    <div class="tt-label">${getNodeLabel(d.source)} → ${getNodeLabel(d.target)}</div>
                    <div class="tt-type">${d.count} mention(s) | ${d.unique_speakers} speaker(s) | conf: ${d.mean_confidence.toFixed(2)}</div>
                `);
            })
            .on("mousemove", (event) => moveTooltip(event))
            .on("mouseleave", hideTooltip)
            .on("click", (event, d) => { hideTooltip(); openEdgeModal(d); });
        linkEnter.merge(linkSel)
            .attr("stroke", "#64748b")
            .attr("stroke-width", d => Math.max(1, Math.min(6, d.unique_speakers * 1.5)))
            .attr("stroke-opacity", d => 0.3 + d.mean_confidence * 0.7)
            .attr("stroke-dasharray", d => d.has_inferred ? "4,3" : null);

        // Edge labels
        const edgeLabelSel = labelGroup.selectAll("text").data(showLabels ? links : [], d => `${d.source.node_id || d.source}|${d.target.node_id || d.target}|${d.relation}`);
        edgeLabelSel.exit().remove();
        edgeLabelSel.enter().append("text").attr("class", "edge-label")
            .merge(edgeLabelSel)
            .text(d => d.relation);

        // Nodes
        const maxWeight = Math.max(1, ...nodeWeights.values());
        const nodeData = nodeArray.filter(n => !hideIsolated || nodeWeights.get(n.node_id) > 0);

        const nodeSel = nodeGroup.selectAll("circle").data(nodeData, d => d.node_id);
        nodeSel.exit().remove();
        const nodeEnter = nodeSel.enter().append("circle")
            .attr("cursor", "pointer")
            .call(d3.drag()
                .on("start", dragstarted)
                .on("drag", dragged)
                .on("end", dragended))
            .on("mouseenter", (event, d) => {
                const w = nodeWeights.get(d.node_id) || 0;
                showTooltip(event, `
                    <div class="tt-label">${d.label}</div>
                    <div class="tt-type">${d.type} | weight: ${w}</div>
                    ${d.aliases.length > 1 ? `<div class="tt-type">aliases: ${d.aliases.join(", ")}</div>` : ""}
                `);
            })
            .on("mousemove", (event) => moveTooltip(event))
            .on("mouseleave", hideTooltip)
            .on("click", (event, d) => { hideTooltip(); openNodeModal(d, nodeWeights.get(d.node_id), filtered, aggEdges); });
        nodeEnter.merge(nodeSel)
            .attr("r", d => {
                const w = nodeWeights.get(d.node_id) || 0;
                return w === 0 ? 4 : 6 + (w / maxWeight) * 20;
            })
            .attr("fill", d => nodeColor(d.type))
            .attr("stroke", "#ffffff")
            .attr("stroke-width", 1.5)
            .attr("opacity", d => nodeWeights.get(d.node_id) > 0 ? 1 : 0.25)
            .on("click", (event, d) => { hideTooltip(); openNodeModal(d, nodeWeights.get(d.node_id), filtered, aggEdges); });

        // Node labels
        const textSel = textGroup.selectAll("text").data(nodeData.filter(d => nodeWeights.get(d.node_id) > 0), d => d.node_id);
        textSel.exit().remove();
        textSel.enter().append("text").attr("class", "node-label").attr("dy", -14)
            .merge(textSel)
            .text(d => d.label.length > 20 ? d.label.slice(0, 18) + "..." : d.label);

        simulation.nodes(nodeData);
        simulation.alpha(0.3).restart();
    }

    simulation.on("tick", () => {
        linkGroup.selectAll("line")
            .attr("x1", d => d.source.x).attr("y1", d => d.source.y)
            .attr("x2", d => d.target.x).attr("y2", d => d.target.y);
        labelGroup.selectAll("text")
            .attr("x", d => (d.source.x + d.target.x) / 2)
            .attr("y", d => (d.source.y + d.target.y) / 2);
        nodeGroup.selectAll("circle")
            .attr("cx", d => d.x).attr("cy", d => d.y);
        textGroup.selectAll("text")
            .attr("x", d => d.x).attr("y", d => d.y);
    });

    function dragstarted(event, d) { if (!event.active) simulation.alphaTarget(0.3).restart(); d.fx = d.x; d.fy = d.y; }
    function dragged(event, d) { d.fx = event.x; d.fy = event.y; }
    function dragended(event, d) { if (!event.active) simulation.alphaTarget(0); d.fx = null; d.fy = null; }

    function nodeColor(type) {
        const colors = {
            actor: "#f97316", technology: "#06b6d4", policy: "#8b5cf6",
            event: "#ef4444", process: "#22c55e", outcome: "#eab308",
            value: "#ec4899", belief_proposition: "#14b8a6", topic: "#6366f1",
        };
        return colors[type] || "#8899a6";
    }

    function openNodeModal(node, weight, filteredEdges, aggEdges) {
        const connected = aggEdges.filter(e => {
            const src = e.source.node_id || e.source;
            const tgt = e.target.node_id || e.target;
            return src === node.node_id || tgt === node.node_id;
        });

        const outgoing = connected.filter(e => (e.source.node_id || e.source) === node.node_id);
        const incoming = connected.filter(e => (e.target.node_id || e.target) === node.node_id);

        let connectionsHtml = "";
        if (outgoing.length > 0) {
            connectionsHtml += `<div class="modal-section"><h4>Outgoing relationships (${outgoing.length})</h4>`;
            outgoing.forEach(e => {
                connectionsHtml += `<div class="modal-evidence"><strong>${e.relation}</strong> → ${getNodeLabel(e.target)}<div class="ev-meta">${e.count} mention(s) | conf: ${e.mean_confidence.toFixed(2)}</div></div>`;
            });
            connectionsHtml += `</div>`;
        }
        if (incoming.length > 0) {
            connectionsHtml += `<div class="modal-section"><h4>Incoming relationships (${incoming.length})</h4>`;
            incoming.forEach(e => {
                connectionsHtml += `<div class="modal-evidence">${getNodeLabel(e.source)} → <strong>${e.relation}</strong><div class="ev-meta">${e.count} mention(s) | conf: ${e.mean_confidence.toFixed(2)}</div></div>`;
            });
            connectionsHtml += `</div>`;
        }

        openModal(`
            <button class="modal-close" onclick="document.getElementById('modal-overlay').classList.remove('active')">&times;</button>
            <h2>${node.label}</h2>
            <div class="modal-subtitle">${node.type}</div>
            <div class="modal-stats">
                <div class="stat-card"><div class="stat-val">${weight || 0}</div><div class="stat-lbl">Mentions</div></div>
                <div class="stat-card"><div class="stat-val">${outgoing.length}</div><div class="stat-lbl">Outgoing</div></div>
                <div class="stat-card"><div class="stat-val">${incoming.length}</div><div class="stat-lbl">Incoming</div></div>
            </div>
            ${node.aliases.length > 1 ? `<div class="modal-section" style="margin-top:16px;"><h4>Aliases</h4><p style="font-size:13px;">${node.aliases.join(", ")}</p></div>` : ""}
            ${connectionsHtml || `<div class="modal-section"><p style="color:#9ca3af;">No connections under current filter.</p></div>`}
        `);
    }

    function openEdgeModal(edge) {
        let evidenceHtml = edge.evidence.map(e => `
            <div class="modal-evidence">
                "${e.text}"
                <div class="ev-meta">${e.speaker} | stance: ${e.stance}</div>
            </div>
        `).join("");

        openModal(`
            <button class="modal-close" onclick="document.getElementById('modal-overlay').classList.remove('active')">&times;</button>
            <h2>${getNodeLabel(edge.source)} → ${getNodeLabel(edge.target)}</h2>
            <div class="modal-subtitle">${edge.relation}</div>
            <div class="modal-stats">
                <div class="stat-card"><div class="stat-val">${edge.count}</div><div class="stat-lbl">Mentions</div></div>
                <div class="stat-card"><div class="stat-val">${edge.unique_speakers}</div><div class="stat-lbl">Speakers</div></div>
                <div class="stat-card"><div class="stat-val">${edge.mean_confidence.toFixed(2)}</div><div class="stat-lbl">Confidence</div></div>
            </div>
            <div class="modal-section" style="margin-top:16px;">
                <span class="modal-badge badge-stance">${edge.has_inferred ? "includes inferred" : "explicit"}</span>
            </div>
            <div class="modal-section">
                <h4>Evidence (${edge.evidence.length})</h4>
                ${evidenceHtml}
            </div>
        `);
    }

    function getNodeLabel(idOrObj) {
        if (typeof idOrObj === "object") return idOrObj.label || idOrObj.node_id;
        const n = nodeMap.get(idOrObj);
        return n ? n.label : idOrObj;
    }

    function populateSelect(id, options) {
        const sel = document.getElementById(id);
        for (const opt of options) {
            const el = document.createElement("option");
            el.value = opt.value;
            el.textContent = opt.label;
            sel.appendChild(el);
        }
    }

    // Bind filter events
    ["filter-round", "filter-table", "filter-speaker", "filter-stance"].forEach(id => {
        document.getElementById(id).addEventListener("change", update);
    });
    document.getElementById("filter-confidence").addEventListener("input", (e) => {
        document.getElementById("conf-val").textContent = e.target.value;
        update();
    });
    ["toggle-inferred", "toggle-labels", "toggle-isolated"].forEach(id => {
        document.getElementById(id).addEventListener("change", update);
    });

    // Initial render
    update();
})();

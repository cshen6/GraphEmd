% A pseudo-code for leiden, not working right now

function [final_partition, final_modularity] = LeidenNotWorking(adjacency_matrix)
    % Initialize with Louvain method
    [initial_partition, initial_modularity] = louvain(adjacency_matrix);
    current_partition = initial_partition;
    current_modularity = initial_modularity;
    
    % Flag to indicate if there is any improvement in modularity
    improved = true;
    
    % Main loop
    while improved
        improved = false;
        
        % Iterate over each node
        for node = 1:size(adjacency_matrix, 1)
            % Calculate the modularity change by moving the node to its neighboring communities
            neighboring_communities = find(adjacency_matrix(node, :) > 0);
            for community = neighboring_communities
                new_partition = current_partition;
                new_partition(node) = community;
                
                new_modularity = calculate_modularity(adjacency_matrix, new_partition);
                
                % If modularity improves, update partition and modularity
                if new_modularity > current_modularity
                    current_partition = new_partition;
                    current_modularity = new_modularity;
                    improved = true;
                end
            end
        end
        
        % Aggregate nodes belonging to the same community
        current_partition = aggregate_nodes(adjacency_matrix, current_partition);
        
        % Recalculate modularity
        current_modularity = calculate_modularity(adjacency_matrix, current_partition);
    end
    
    % Output final partition and modularity
    final_partition = current_partition;
    final_modularity = current_modularity;
end

function modularity = calculate_modularity(adjacency_matrix, partition)
    % Calculate modularity based on the given partition
    total_edges = sum(adjacency_matrix(:)) / 2;
    modularity = 0;
    num_nodes = size(adjacency_matrix, 1);
    for i = 1:num_nodes
        for j = 1:num_nodes
            modularity = modularity + (adjacency_matrix(i, j) - ((sum(adjacency_matrix(i, :)) * sum(adjacency_matrix(:, j))) / (2 * total_edges))) * (partition(i) == partition(j));
        end
    end
    modularity = modularity / (2 * total_edges);
end

function new_partition = aggregate_nodes(adjacency_matrix, partition)
    % Aggregate nodes belonging to the same community
    num_nodes = size(adjacency_matrix, 1);
    unique_communities = unique(partition);
    new_partition = zeros(1, num_nodes);
    for i = 1:length(unique_communities)
        community = unique_communities(i);
        nodes_in_community = find(partition == community);
        new_partition(nodes_in_community) = i;
    end
end

function [partition, modularity] = louvain(adjacency_matrix)
    % Initialize each node to its own community
    partition = 1:size(adjacency_matrix, 1);
    
    % Initial modularity
    modularity = calculate_modularity(adjacency_matrix, partition);
    
    % Flag to indicate if there is any improvement in modularity
    improved = true;
    
    % Main loop
    while improved
        improved = false;
        
        % Iterate over each node
        for node = 1:size(adjacency_matrix, 1)
            % Calculate the change in modularity by moving the node to its neighboring communities
            neighboring_communities = unique(partition(find(adjacency_matrix(node, :))));
            best_community = partition(node);
            best_modularity = modularity;
            for community = neighboring_communities
                new_partition = partition;
                new_partition(node) = community;
                
                new_modularity = calculate_modularity(adjacency_matrix, new_partition);
                
                % If modularity improves, update partition and modularity
                if new_modularity > best_modularity
                    best_community = community;
                    best_modularity = new_modularity;
                    improved = true;
                end
            end
            
            % Update partition if there is improvement
            if best_community ~= partition(node)
                partition(node) = best_community;
                modularity = best_modularity;
            end
        end
    end
end